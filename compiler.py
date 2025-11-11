import torch
import torch.nn as nn
import numpy as np
import random
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any, Dict
import heapq

# Scipy and Scikit-learn imports
from scipy.optimize import linprog

# TQDM for progress bars
from tqdm import tqdm


# --- 共同设置 (Common Settings) ---
SEED = 2025
FP_TOLERANCE = 1e-5  # 相等阈值 τ

# 设置随机种子以保证可复现性
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# --- §2. 核心数据结构 (Core Data Structures) ---
# =============================================================================

@dataclass
class Guard:
    """ 定义一个由 Ax <= d 描述的凸多面体区域，并提供基本操作 """
    A: np.ndarray
    d: np.ndarray

    def __post_init__(self):
        if self.A.ndim != 2:
            raise ValueError("Guard.A 必须为二维矩阵")
        if self.d.ndim != 1:
            raise ValueError("Guard.d 必须为一维向量")
        if self.A.shape[0] != self.d.shape[0]:
            raise ValueError("Guard 的不等式数量必须匹配")

    @property
    def dimension(self) -> int:
        return self.A.shape[1] if self.A.ndim == 2 else 0

    def copy(self) -> 'Guard':
        return Guard(A=self.A.copy(), d=self.d.copy())

    def intersect(self, other: 'Guard') -> 'Guard':
        if self.A.size == 0:
            return other.copy()
        if other.A.size == 0:
            return self.copy()
        if self.dimension != other.dimension:
            raise ValueError("维度不匹配，无法计算守卫的交集")
        new_A = np.vstack([self.A, other.A]) if self.A.size else other.A.copy()
        new_d = np.concatenate([self.d, other.d]) if self.d.size else other.d.copy()
        return Guard(A=new_A, d=new_d)

    def append_constraint(self, a: np.ndarray, b: float) -> 'Guard':
        if a.ndim != 1:
            raise ValueError("约束向量必须是一维")
        if self.dimension not in (0, a.shape[0]):
            raise ValueError("约束维度与 Guard 不匹配")
        if self.dimension == 0:
            A = a.reshape(1, -1)
        else:
            A = np.vstack([self.A, a.reshape(1, -1)])
        d = np.concatenate([self.d, np.array([b])]) if self.d.size else np.array([b])
        return Guard(A=A, d=d)

    def is_feasible(self) -> bool:
        if self.A.size == 0:
            return True
        bounds = [(None, None)] * self.dimension
        c = np.zeros(self.dimension)
        res = linprog(c, A_ub=self.A, b_ub=self.d, bounds=bounds, method='highs')
        return res.success

    def bounds_for_linear(self, w: np.ndarray, b: float = 0.0) -> Tuple[float, float]:
        if self.A.size == 0:
            return -np.inf, np.inf
        if w.ndim != 1 or w.shape[0] != self.dimension:
            raise ValueError("线性函数的维度与 Guard 不匹配")
        bounds = [(None, None)] * self.dimension
        res_min = linprog(w, A_ub=self.A, b_ub=self.d, bounds=bounds, method='highs')
        res_max = linprog(-w, A_ub=self.A, b_ub=self.d, bounds=bounds, method='highs')
        if not (res_min.success and res_max.success):
            return -np.inf, np.inf
        return res_min.fun + b, -res_max.fun + b

@dataclass
class Piece:
    """ 将一个守卫(Guard)与一个仿射变换(weights, bias)绑定 """
    guard: Guard
    weights: np.ndarray
    bias: np.ndarray
    exact: bool = True
    metadata: dict = field(default_factory=dict)

    def clone(self, *, exact: Optional[bool] = None) -> 'Piece':
        return Piece(
            guard=self.guard.copy(),
            weights=self.weights.copy(),
            bias=self.bias.copy(),
            exact=self.exact if exact is None else exact,
            metadata=self.metadata.copy()
        )

# =============================================================================
# --- §3. 自动机核心组件 (Automaton Core Components) ---
# =============================================================================

@dataclass
class State:
    """ 代表自动机的一个状态 (Q)，通常对应于网络层之间 """
    id: Any

@dataclass
class CompilationState:
    """ 在动态编译过程中携带和传递状态 """
    piece: Piece
    current_x: torch.Tensor
    current_shape: Optional[Tuple]
    pre_activation: torch.Tensor
    adj_matrix: Optional[np.ndarray] = None


@dataclass
class SymbolicContext:
    shape: Optional[Tuple]
    pre_activation: Optional[np.ndarray] = None
    adj_matrix: Optional[np.ndarray] = None

class SymbolicTransition:
    """
    代表自动机的转移函数 δ。
    这是一个抽象基类，封装了单层网络的符号化逻辑。
    """
    def __init__(self, layer: nn.Module, from_state: State, to_state: State):
        self.layer = layer
        self.from_state = from_state
        self.to_state = to_state

    def apply(self, state: CompilationState) -> CompilationState:
        raise NotImplementedError

    def transform_piece(self, piece: Piece, ctx: SymbolicContext,
                        branch_selector=None) -> List[Piece]:
        raise NotImplementedError

# --- 具体转移的实现 ---

class LinearTransition(SymbolicTransition):
    def apply(self, state: CompilationState) -> CompilationState:
        W = self.layer.weight.cpu().detach().numpy()
        b = self.layer.bias.cpu().detach().numpy()

        is_nodewise = state.current_shape is not None and len(state.current_shape) == 2 and state.current_shape[0] > 1 and state.current_shape[1] == self.layer.in_features

        if is_nodewise:
            N, D_in = state.current_shape
            D_out = self.layer.out_features
            W_kron = np.kron(np.eye(N), W)
            b_tiled = np.tile(b, N)
            state.piece.weights = W_kron @ state.piece.weights
            state.piece.bias = W_kron @ state.piece.bias + b_tiled
            state.current_shape = (N, D_out)
        else:
            state.piece.weights = W @ state.piece.weights
            state.piece.bias = W @ state.piece.bias + b
            if state.current_shape and len(state.current_shape) == 1:
                state.current_shape = (self.layer.out_features,)
        return state

    def transform_piece(self, piece: Piece, ctx: SymbolicContext, branch_selector=None) -> List[Piece]:
        W = self.layer.weight.cpu().detach().numpy()
        b = self.layer.bias.cpu().detach().numpy()
        new_piece = piece.clone()
        if ctx.shape is not None and len(ctx.shape) == 2 and ctx.shape[0] > 1 and ctx.shape[1] == self.layer.in_features:
            N, D_in = ctx.shape
            D_out = self.layer.out_features
            W_kron = np.kron(np.eye(N), W)
            b_tiled = np.tile(b, N)
            new_piece.weights = W_kron @ new_piece.weights
            new_piece.bias = W_kron @ new_piece.bias + b_tiled
            ctx.shape = (N, D_out)
        else:
            new_piece.weights = W @ new_piece.weights
            new_piece.bias = W @ new_piece.bias + b
            if ctx.shape and len(ctx.shape) == 1:
                ctx.shape = (self.layer.out_features,)
        return [new_piece]

class ReLuTransition(SymbolicTransition):
    def apply(self, state: CompilationState) -> CompilationState:
        pre_act_flat = state.pre_activation.cpu().numpy().flatten()
        active_map = pre_act_flat > FP_TOLERANCE

        new_guards_A, new_guards_d = [], []
        for i, is_active in enumerate(active_map):
            w_i, b_i = state.piece.weights[i], state.piece.bias[i]
            if is_active:
                new_guards_A.append(-w_i)
                new_guards_d.append(b_i)
            else:
                new_guards_A.append(w_i)
                new_guards_d.append(-b_i)

        state.piece.weights[~active_map, :] = 0
        state.piece.bias[~active_map] = 0

        if new_guards_A:
            A = np.array(new_guards_A, dtype=float)
            if A.ndim == 1:
                A = A.reshape(1, -1)
            d = np.array(new_guards_d, dtype=float)
            state.piece.guard = state.piece.guard.intersect(Guard(A, d))
        return state

    def transform_piece(self, piece: Piece, ctx: SymbolicContext, branch_selector=None) -> List[Piece]:
        pre_act = ctx.pre_activation
        output_dim = piece.bias.shape[0]
        if pre_act is not None:
            pre_act_flat = pre_act.flatten()
        else:
            pre_act_flat = None

        resulting_pieces = [piece.clone()]

        for idx in range(output_dim):
            next_pieces = []
            for current_piece in resulting_pieces:
                w_i = current_piece.weights[idx]
                b_i = current_piece.bias[idx]
                guard = current_piece.guard
                lb, ub = guard.bounds_for_linear(w_i, b_i)

                if lb >= -FP_TOLERANCE:
                    next_pieces.append(current_piece)
                    continue
                if ub <= FP_TOLERANCE:
                    inactive_piece = current_piece.clone()
                    inactive_piece.weights[idx, :] = 0
                    inactive_piece.bias[idx] = 0
                    next_pieces.append(inactive_piece)
                    continue

                decision = None
                if branch_selector is not None:
                    decision = branch_selector(
                        {
                            'type': 'relu',
                            'index': idx,
                            'lower': lb,
                            'upper': ub,
                            'pre_activation': None if pre_act_flat is None else pre_act_flat[idx]
                        }
                    )

                if decision == 'positive':
                    pos_piece = current_piece.clone()
                    pos_piece.guard = pos_piece.guard.append_constraint(-w_i, b_i)
                    next_pieces.append(pos_piece)
                elif decision == 'negative':
                    neg_piece = current_piece.clone()
                    neg_piece.guard = neg_piece.guard.append_constraint(w_i, -b_i)
                    neg_piece.weights[idx, :] = 0
                    neg_piece.bias[idx] = 0
                    next_pieces.append(neg_piece)
                else:
                    pos_piece = current_piece.clone()
                    pos_piece.guard = pos_piece.guard.append_constraint(-w_i, b_i)
                    neg_piece = current_piece.clone()
                    neg_piece.guard = neg_piece.guard.append_constraint(w_i, -b_i)
                    neg_piece.weights[idx, :] = 0
                    neg_piece.bias[idx] = 0
                    for candidate in (pos_piece, neg_piece):
                        if candidate.guard.is_feasible():
                            next_pieces.append(candidate)
            resulting_pieces = next_pieces

        for result_piece in resulting_pieces:
            result_piece.metadata['post_relu_shape'] = ctx.shape

        return resulting_pieces

class FlattenTransition(SymbolicTransition):
    def apply(self, state: CompilationState) -> CompilationState:
        if state.current_shape:
            state.current_shape = (np.prod(state.current_shape),)
        return state

    def transform_piece(self, piece: Piece, ctx: SymbolicContext, branch_selector=None) -> List[Piece]:
        if ctx.shape is not None:
            ctx.shape = (int(np.prod(ctx.shape)),)
        return [piece.clone()]

class AdaptiveAvgPool2dTransition(SymbolicTransition):
    def apply(self, state: CompilationState) -> CompilationState:
        if self.layer.output_size == 1 or self.layer.output_size == (1, 1):
            C, H, W = state.current_shape
            W_pool = np.zeros((C, C * H * W))
            avg_val = 1.0 / (H * W)
            for c in range(C):
                start_idx, end_idx = c * H * W, (c + 1) * H * W
                W_pool[c, start_idx:end_idx] = avg_val
            
            state.piece.weights = W_pool @ state.piece.weights
            state.piece.bias = W_pool @ state.piece.bias
            state.current_shape = (C,)
        return state

    def transform_piece(self, piece: Piece, ctx: SymbolicContext, branch_selector=None) -> List[Piece]:
        new_piece = piece.clone()
        if self.layer.output_size == 1 or self.layer.output_size == (1, 1):
            if ctx.shape is None:
                raise ValueError("AdaptiveAvgPool2d 需要已知的输入形状")
            C, H, W = ctx.shape
            W_pool = np.zeros((C, C * H * W))
            avg_val = 1.0 / (H * W)
            for c in range(C):
                start_idx, end_idx = c * H * W, (c + 1) * H * W
                W_pool[c, start_idx:end_idx] = avg_val
            new_piece.weights = W_pool @ new_piece.weights
            new_piece.bias = W_pool @ new_piece.bias
            ctx.shape = (C,)
        return [new_piece]

class Conv2dTransition(SymbolicTransition):
    def apply(self, state: CompilationState) -> CompilationState:
        C_in, H_in, W_in = state.current_shape
        C_out, _, k_h, k_w = self.layer.weight.shape
        s_h, s_w = self.layer.stride
        p_h, p_w = self.layer.padding
        H_out = (H_in + 2 * p_h - k_h) // s_h + 1
        W_out = (W_in + 2 * p_w - k_w) // s_w + 1
        
        W_kernel = self.layer.weight.cpu().detach().numpy()
        b_kernel = self.layer.bias.cpu().detach().numpy()
        W_matrix = np.zeros((C_out * H_out * W_out, C_in * H_in * W_in))
        
        for c_out, h_out, w_out in np.ndindex(C_out, H_out, W_out):
            out_idx = c_out * H_out * W_out + h_out * W_out + w_out
            h_start, w_start = h_out * s_h - p_h, w_out * s_w - p_w
            for c_in, kh, kw in np.ndindex(C_in, k_h, k_w):
                h_in, w_in = h_start + kh, w_start + kw
                if 0 <= h_in < H_in and 0 <= w_in < W_in:
                    in_idx = c_in * H_in * W_in + h_in * W_in + w_in
                    W_matrix[out_idx, in_idx] = W_kernel[c_out, c_in, kh, kw]
        
        b_vector = np.repeat(b_kernel, H_out * W_out)
        state.piece.weights = W_matrix @ state.piece.weights
        state.piece.bias = W_matrix @ state.piece.bias + b_vector
        state.current_shape = (C_out, H_out, W_out)
        return state

    def transform_piece(self, piece: Piece, ctx: SymbolicContext, branch_selector=None) -> List[Piece]:
        if ctx.shape is None:
            raise ValueError("Conv2d 需要上下文形状信息")
        C_in, H_in, W_in = ctx.shape
        C_out, _, k_h, k_w = self.layer.weight.shape
        s_h, s_w = self.layer.stride
        p_h, p_w = self.layer.padding
        H_out = (H_in + 2 * p_h - k_h) // s_h + 1
        W_out = (W_in + 2 * p_w - k_w) // s_w + 1

        W_kernel = self.layer.weight.cpu().detach().numpy()
        b_kernel = self.layer.bias.cpu().detach().numpy()
        W_matrix = np.zeros((C_out * H_out * W_out, C_in * H_in * W_in))

        for c_out, h_out, w_out in np.ndindex(C_out, H_out, W_out):
            out_idx = c_out * H_out * W_out + h_out * W_out + w_out
            h_start, w_start = h_out * s_h - p_h, w_out * s_w - p_w
            for c_in, kh, kw in np.ndindex(C_in, k_h, k_w):
                h_in, w_in = h_start + kh, w_start + kw
                if 0 <= h_in < H_in and 0 <= w_in < W_in:
                    in_idx = c_in * H_in * W_in + h_in * W_in + w_in
                    W_matrix[out_idx, in_idx] = W_kernel[c_out, c_in, kh, kw]

        b_vector = np.repeat(b_kernel, H_out * W_out)

        new_piece = piece.clone()
        new_piece.weights = W_matrix @ new_piece.weights
        new_piece.bias = W_matrix @ new_piece.bias + b_vector
        ctx.shape = (C_out, H_out, W_out)
        return [new_piece]

class GCNConvTransition(SymbolicTransition):
    def apply(self, state: CompilationState) -> CompilationState:
        N, D_in = state.current_shape
        D_out = self.layer.linear.out_features
        input_dim_flat, output_dim_flat = N * D_in, N * D_out
        W_gcn = np.zeros((output_dim_flat, input_dim_flat))

        original_bias = self.layer.linear.bias
        self.layer.linear.bias = None
        adj_t = torch.tensor(state.adj_matrix, dtype=torch.float32).to(DEVICE)

        for i in range(input_dim_flat):
            e_i = torch.zeros(input_dim_flat, device=DEVICE)
            e_i[i] = 1.0
            e_i_mat = e_i.reshape((N, D_in))
            y_i = self.layer(e_i_mat, adj_t)
            W_gcn[:, i] = y_i.cpu().numpy().flatten()
        
        self.layer.linear.bias = original_bias
        b = original_bias.cpu().numpy()
        b_gcn = np.tile(b, N)
        
        state.piece.weights = W_gcn @ state.piece.weights
        state.piece.bias = W_gcn @ state.piece.bias + b_gcn
        state.current_shape = (N, D_out)
        return state

    def transform_piece(self, piece: Piece, ctx: SymbolicContext, branch_selector=None) -> List[Piece]:
        if ctx.shape is None:
            raise ValueError("GCNConv 需要上下文形状信息")
        if ctx.adj_matrix is None:
            raise ValueError("GCNConv 需要提供邻接矩阵")
        N, D_in = ctx.shape
        D_out = self.layer.linear.out_features
        input_dim_flat, output_dim_flat = N * D_in, N * D_out
        W_gcn = np.zeros((output_dim_flat, input_dim_flat))

        original_bias = self.layer.linear.bias
        self.layer.linear.bias = None
        adj_t = torch.tensor(ctx.adj_matrix, dtype=torch.float32).to(DEVICE)

        for i in range(input_dim_flat):
            e_i = torch.zeros(input_dim_flat, device=DEVICE)
            e_i[i] = 1.0
            e_i_mat = e_i.reshape((N, D_in))
            y_i = self.layer(e_i_mat, adj_t)
            W_gcn[:, i] = y_i.cpu().numpy().flatten()

        self.layer.linear.bias = original_bias
        b = original_bias.cpu().numpy() if original_bias is not None else np.zeros(D_out)
        b_gcn = np.tile(b, N)

        new_piece = piece.clone()
        new_piece.weights = W_gcn @ new_piece.weights
        new_piece.bias = W_gcn @ new_piece.bias + b_gcn
        ctx.shape = (N, D_out)
        return [new_piece]
        
class InterventionTransition(SymbolicTransition):
    def apply(self, state: CompilationState) -> CompilationState:
        N = self.layer.num_nodes
        D_h = self.layer.hidden_dim
        channel_to_zero = self.layer.channel
        
        M = np.eye(N * D_h)
        
        for node_idx in range(N):
            flat_idx = node_idx * D_h + channel_to_zero
            if flat_idx < M.shape[0]:
                M[flat_idx, flat_idx] = 0
        
        state.piece.weights = M @ state.piece.weights
        state.piece.bias = M @ state.piece.bias

        return state

    def transform_piece(self, piece: Piece, ctx: SymbolicContext, branch_selector=None) -> List[Piece]:
        N = self.layer.num_nodes
        D_h = self.layer.hidden_dim
        channel_to_zero = self.layer.channel

        M = np.eye(N * D_h)
        for node_idx in range(N):
            flat_idx = node_idx * D_h + channel_to_zero
            if flat_idx < M.shape[0]:
                M[flat_idx, flat_idx] = 0

        new_piece = piece.clone()
        new_piece.weights = M @ new_piece.weights
        new_piece.bias = M @ new_piece.bias
        return [new_piece]

# =============================================================================
# --- §3.6 Imax计算的特定模型与数据结构 (Imax Specific Models & Datastructures) ---
# =============================================================================

@dataclass(order=True)
class BBNode:
    priority: float = field(init=False)
    upper_bound: float
    guard: Guard = field(compare=False)
    
    def __post_init__(self):
        self.priority = -self.upper_bound
        
# =============================================================================
# --- §4. 加权自动机 (Weighted Automaton) ---
# =============================================================================

class WeightedAutomaton:
    def __init__(self, model: nn.Module):
        self.model = model
        self.states: List[State] = []
        self.transitions: List[SymbolicTransition] = []
        self._build()

    def _build(self):
        self.states.append(State(id="input"))
        # Dynamically import GCNConv and Intervention only when needed to avoid circular dependencies
        from torch.nn import Linear, ReLU, Flatten, AdaptiveAvgPool2d, Conv2d
        
        # This approach is tricky because exp3 models are now defined in exp3.py.
        # A better approach is to check type names.
        
        for i, layer in enumerate(self.model.children()):
            from_state = self.states[-1]
            to_state = State(id=f"layer_{i}_{type(layer).__name__}")
            
            layer_type_name = type(layer).__name__

            if layer_type_name == 'Linear':
                transition = LinearTransition(layer, from_state, to_state)
            elif layer_type_name == 'ReLU':
                transition = ReLuTransition(layer, from_state, to_state)
            elif layer_type_name == 'Flatten':
                transition = FlattenTransition(layer, from_state, to_state)
            elif layer_type_name == 'AdaptiveAvgPool2d':
                transition = AdaptiveAvgPool2dTransition(layer, from_state, to_state)
            elif layer_type_name == 'Conv2d':
                transition = Conv2dTransition(layer, from_state, to_state)
            elif layer_type_name == 'GCNConv':
                transition = GCNConvTransition(layer, from_state, to_state)
            elif layer_type_name == 'Intervention':
                transition = InterventionTransition(layer, from_state, to_state)
            else:
                continue
            
            self.transitions.append(transition)
            self.states.append(to_state)

    def trace(self, x_np: np.ndarray, initial_piece: Piece, initial_shape: Optional[Tuple], adj_matrix: Optional[np.ndarray]) -> Piece:
        x_tensor = torch.tensor(x_np, dtype=torch.float32).to(DEVICE)
        current_x_flat = x_tensor.flatten().unsqueeze(0)

        state = CompilationState(
            piece=initial_piece,
            current_x=current_x_flat,
            current_shape=initial_shape,
            pre_activation=current_x_flat,
            adj_matrix=adj_matrix
        )

        with torch.no_grad():
            for transition in self.transitions:
                state.pre_activation = state.current_x
                
                # The check here should also be by name or a more robust method
                # 之前的代码直接使用了 type(transition.layer).__name__
                # 我们将其提取到一个变量中，以供后续的if/elif/else块使用
                layer_type_name = type(transition.layer).__name__
                
                if layer_type_name == 'GCNConv':
                    reshaped_x = state.current_x.view(state.current_shape)
                    adj_t = torch.tensor(state.adj_matrix, dtype=torch.float32).to(DEVICE)
                    output = transition.layer(reshaped_x, adj_t)
                    state.current_x = output
                elif layer_type_name in ['Conv2d', 'AdaptiveAvgPool2d']:
                    # 关键修复：在进入卷积或池化层前，将扁平的张量还原为图像格式
                    # 形状应为 (批大小, 通道数, 高, 宽)，此处的批大小为 1
                    reshaped_x = state.current_x.view(1, *state.current_shape)
                    state.current_x = transition.layer(reshaped_x)
                else:
                    if len(state.current_x.shape) > 2 and not isinstance(transition.layer, (nn.Conv2d, nn.AdaptiveAvgPool2d, nn.Flatten)):
                         state.current_x = state.current_x.view(state.current_x.size(0), -1)
                    state.current_x = transition.layer(state.current_x)

                state = transition.apply(state)

        return state.piece

    def propagate_symbolic(self, initial_piece: Piece, initial_shape: Optional[Tuple],
                           adj_matrix: Optional[np.ndarray] = None,
                           branch_selector=None,
                           pre_activations: Optional[List[np.ndarray]] = None) -> List[Piece]:
        pieces_with_ctx: List[Tuple[Piece, SymbolicContext]] = [
            (initial_piece.clone(), SymbolicContext(shape=None if initial_shape is None else tuple(initial_shape),
                                                    adj_matrix=adj_matrix))
        ]

        pre_activation_iter = iter(pre_activations or [])

        for transition in self.transitions:
            next_collection: List[Tuple[Piece, SymbolicContext]] = []
            layer_name = type(transition.layer).__name__
            for piece, ctx in pieces_with_ctx:
                if layer_name == 'ReLU':
                    ctx.pre_activation = next(pre_activation_iter, None)
                else:
                    ctx.pre_activation = None

                local_ctx = SymbolicContext(shape=None if ctx.shape is None else tuple(ctx.shape),
                                            pre_activation=ctx.pre_activation,
                                            adj_matrix=ctx.adj_matrix)
                transformed_pieces = transition.transform_piece(piece, local_ctx, branch_selector)
                for transformed_piece in transformed_pieces:
                    next_ctx = SymbolicContext(shape=None if local_ctx.shape is None else tuple(local_ctx.shape),
                                                adj_matrix=local_ctx.adj_matrix)
                    next_collection.append((transformed_piece, next_ctx))
            pieces_with_ctx = next_collection

        return [p for p, _ in pieces_with_ctx]

class SWT_Compiler:
    def __init__(self, model: nn.Module, input_dim: int,
                 input_domain: Optional[Guard] = None,
                 initial_shape: Optional[Tuple] = None,
                 adj_matrix: Optional[torch.Tensor] = None):
        self.model = model.to(DEVICE).eval()
        self.input_dim = input_dim
        self.initial_shape = initial_shape
        self.adj_matrix = adj_matrix.cpu().numpy() if adj_matrix is not None else None

        if input_domain is None:
            self.input_domain = Guard(A=np.empty((0, input_dim)), d=np.empty(0))
        else:
            self.input_domain = input_domain.copy()

        self.automaton = WeightedAutomaton(self.model)

    def compile_for_input(self, x_np: np.ndarray) -> Optional[Piece]:
        if x_np.ndim == 3 and self.initial_shape is not None and len(self.initial_shape) == 3:
            x_tensor = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        else:
            x_tensor = torch.tensor(x_np.flatten(), dtype=torch.float32).unsqueeze(0).to(DEVICE)

        pre_activations = self._collect_pre_activations(x_tensor)

        def local_branch_selector(info: Dict[str, Any]):
            if info['type'] != 'relu':
                return None
            pre_val = info.get('pre_activation')
            if pre_val is None:
                return None
            if pre_val > FP_TOLERANCE:
                return 'positive'
            if pre_val < -FP_TOLERANCE:
                return 'negative'
            return None

        initial_piece = Piece(
            guard=self.input_domain.copy(),
            weights=np.eye(self.input_dim),
            bias=np.zeros(self.input_dim)
        )

        pieces = self.automaton.propagate_symbolic(
            initial_piece,
            initial_shape=self.initial_shape,
            adj_matrix=self.adj_matrix,
            branch_selector=local_branch_selector,
            pre_activations=pre_activations
        )

        if not pieces:
            return None
        if len(pieces) > 1:
            # 由于局部分支选择不唯一，保留第一个精确片段
            return pieces[0]
        return pieces[0]

    def get_lipschitz_upper_bound(self):
        L_upper = 1.0
        with torch.no_grad():
            for m in self.model.children():
                # Check by name to avoid direct dependency
                if type(m).__name__ == 'Linear':
                    L_upper *= torch.linalg.matrix_norm(m.weight, ord=2).item()
                elif type(m).__name__ == 'GCNConv':
                    L_upper *= torch.linalg.matrix_norm(m.linear.weight, ord=2).item()
        return L_upper

    # ----------------- 新增：公共工具与审计/区域编译 -----------------

    class _BudgetedBranchSelector:
        def __init__(self, split_budget: int):
            self.split_budget = split_budget
            self.performed_splits = 0

        def __call__(self, info: Dict[str, Any]):
            if info['type'] != 'relu':
                return None
            if self.split_budget < 0:
                return None
            if self.performed_splits >= self.split_budget:
                lower, upper = info['lower'], info['upper']
                if abs(lower) <= abs(upper):
                    return 'positive'
                return 'negative'
            self.performed_splits += 1
            return None

    def compile_guard(self, guard: Guard, split_budget: int = -1) -> List[Piece]:
        selector = None if split_budget < 0 else self._BudgetedBranchSelector(split_budget)
        initial_piece = Piece(
            guard=guard.copy(),
            weights=np.eye(self.input_dim),
            bias=np.zeros(self.input_dim)
        )
        pieces = self.automaton.propagate_symbolic(
            initial_piece,
            initial_shape=self.initial_shape,
            adj_matrix=self.adj_matrix,
            branch_selector=selector
        )
        feasible_pieces = []
        for piece in pieces:
            if piece.guard.is_feasible():
                feasible_pieces.append(piece)
        return feasible_pieces

    def _collect_pre_activations(self, x_tensor: torch.Tensor) -> List[np.ndarray]:
        with torch.no_grad():
            current = x_tensor
            pre_acts: List[np.ndarray] = []
            for layer in self.model.children():
                layer_name = type(layer).__name__
                if layer_name in ['Conv2d', 'AdaptiveAvgPool2d']:
                    if current.dim() == 2 and self.initial_shape is not None:
                        current = current.view(1, *self.initial_shape)
                    current = layer(current)
                else:
                    if current.dim() > 2 and layer_name not in ['Flatten']:
                        current = current.view(current.size(0), -1)
                    if layer_name == 'ReLU':
                        pre_acts.append(current.detach().cpu().numpy().copy())
                    current = layer(current)
            return [pa.flatten() for pa in pre_acts]

    # ----------------- 审计相关方法 -----------------

    def audit(self, num_samples: int = 16, guard: Optional[Guard] = None) -> CompilationAuditReport:
        target_guard = guard if guard is not None else self.input_domain
        samples = self._sample_guard(target_guard, num_samples)
        metrics: List[AuditMetric] = []
        failures: List[str] = []

        errors = []
        infeasible = 0
        exact_count = 0

        for idx, sample in enumerate(samples):
            piece = self.compile_for_input(sample)
            if piece is None:
                failures.append(f"sample_{idx}: compilation returned None")
                continue
            if not piece.guard.is_feasible():
                infeasible += 1
            if piece.exact:
                exact_count += 1

            torch_input = self._tensorize(sample)
            with torch.no_grad():
                model_eval = self.model(torch_input).cpu().numpy().reshape(-1)
            piece_eval = piece.weights @ sample.flatten() + piece.bias
            errors.append(float(np.max(np.abs(model_eval - piece_eval))))

        if errors:
            metrics.append(AuditMetric("max_forward_error", max(errors)))
            metrics.append(AuditMetric("mean_forward_error", float(np.mean(errors))))
        else:
            metrics.append(AuditMetric("max_forward_error", float('nan')))
            metrics.append(AuditMetric("mean_forward_error", float('nan')))

        metrics.append(AuditMetric("infeasible_piece_count", infeasible))
        metrics.append(AuditMetric("exact_piece_ratio", exact_count / max(len(samples), 1)))
        metrics.append(AuditMetric("evaluated_samples", len(errors)))
        metrics.append(AuditMetric("compilation_failures", len(failures)))

        return CompilationAuditReport(metrics=metrics, failures=failures)

    def _tensorize(self, x: np.ndarray) -> torch.Tensor:
        if self.initial_shape is not None and len(self.initial_shape) == 3:
            return torch.tensor(x.reshape((1, *self.initial_shape)), dtype=torch.float32).to(DEVICE)
        return torch.tensor(x.reshape(1, -1), dtype=torch.float32).to(DEVICE)

    def _sample_guard(self, guard: Guard, num_samples: int) -> List[np.ndarray]:
        dim = self.input_dim
        bounds = self._guard_bounds(guard)
        samples = []
        center = self._guard_center(guard)
        for _ in range(num_samples):
            point = np.array([np.random.uniform(low, high) for (low, high) in bounds])
            if guard.A.size and not np.all(guard.A @ point <= guard.d + FP_TOLERANCE):
                # interpolate towards center to enforce feasibility
                lam = np.random.uniform(0.0, 1.0)
                point = lam * point + (1 - lam) * center
            samples.append(point)
        return samples

    def _guard_bounds(self, guard: Guard) -> List[Tuple[float, float]]:
        dim = self.input_dim
        if guard.A.size == 0:
            return [(-1.0, 1.0) for _ in range(dim)]
        bounds = []
        base_bounds = [(None, None)] * dim
        for idx in range(dim):
            e = np.zeros(dim)
            e[idx] = 1.0
            res_min = linprog(e, A_ub=guard.A, b_ub=guard.d, bounds=base_bounds, method='highs')
            res_max = linprog(-e, A_ub=guard.A, b_ub=guard.d, bounds=base_bounds, method='highs')
            if res_min.success and res_max.success:
                bounds.append((res_min.fun, -res_max.fun))
            else:
                bounds.append((-1.0, 1.0))
        return bounds

    def _guard_center(self, guard: Guard) -> np.ndarray:
        if guard.A.size == 0:
            return np.zeros(self.input_dim)
        bounds = [(None, None)] * self.input_dim
        c = np.zeros(self.input_dim)
        res = linprog(c, A_ub=guard.A, b_ub=guard.d, bounds=bounds, method='highs')
        if res.success:
            return res.x
        return np.zeros(self.input_dim)

# =============================================================================
# --- §4.5 Imax分支定界求解器 (Imax Branch & Bound Solver) ---
# =============================================================================

class BranchAndBoundSolver:
    def __init__(self, compiler_F: SWT_Compiler, compiler_C: SWT_Compiler, initial_guard: Guard):
        self.compiler_F = compiler_F
        self.compiler_C = compiler_C
        self.initial_guard = initial_guard
        self.input_dim = compiler_F.input_dim
        
    def _get_bounds_in_guard(self, guard: Guard) -> Tuple[float, float]:
        c = np.zeros(self.input_dim)
        bounds = [(None, None)] * self.input_dim
        res = linprog(c, A_ub=guard.A, b_ub=guard.d, bounds=bounds, method='highs')
        
        if not res.success:
            return 0.0, 0.0

        center_point = res.x
        piece_F = self.compiler_F.compile_for_input(center_point)
        piece_C = self.compiler_C.compile_for_input(center_point)
        
        if piece_F is None or piece_C is None:
             return 0.0, float('inf')

        diff_weights = piece_F.weights - piece_C.weights
        diff_bias = piece_F.bias - piece_C.bias
        diff_at_center = np.abs(diff_weights @ center_point + diff_bias)
        lower_bound = np.max(diff_at_center)
        
        upper_bound = 0.0
        for i in range(diff_weights.shape[0]):
            res_max = linprog(-diff_weights[i], A_ub=guard.A, b_ub=guard.d, bounds=bounds, method='highs')
            res_min = linprog(diff_weights[i], A_ub=guard.A, b_ub=guard.d, bounds=bounds, method='highs')
            
            if res_max.success and res_min.success:
                max_val = -res_max.fun + diff_bias[i]
                min_val = res_min.fun + diff_bias[i]
                upper_bound = max(upper_bound, abs(max_val), abs(min_val))
            else:
                return lower_bound, float('inf')
                
        return lower_bound, upper_bound

    def solve(self, max_iter: int = 200, tolerance: float = 1e-4) -> Tuple[float, List]:
        pq = [BBNode(upper_bound=float('inf'), guard=self.initial_guard)]
        heapq.heapify(pq)
        
        global_lower_bound = 0.0
        history = []
        
        pbar = tqdm(total=max_iter, desc="B&B for Imax", leave=False)
        for i in range(max_iter):
            if not pq:
                pbar.close()
                print("B&B queue empty, search complete.")
                break
            
            current_upper_bound = -pq[0].priority if pq else global_lower_bound
            history.append((i, global_lower_bound, current_upper_bound))
            
            node = heapq.heappop(pq)
            pbar.update(1)
            
            if node.upper_bound < global_lower_bound + tolerance:
                continue
            
            lb, ub = self._get_bounds_in_guard(node.guard)
            global_lower_bound = max(global_lower_bound, lb)
            
            if ub < global_lower_bound + tolerance:
                continue
                
            ranges = []
            bounds_per_dim = []
            for dim in range(self.input_dim):
                c = np.zeros(self.input_dim); c[dim] = 1.0
                unconstrained = [(None, None)] * self.input_dim
                res_min = linprog(c, A_ub=node.guard.A, b_ub=node.guard.d, bounds=unconstrained)
                res_max = linprog(-c, A_ub=node.guard.A, b_ub=node.guard.d, bounds=unconstrained)
                if res_min.success and res_max.success:
                    min_val, max_val = res_min.fun, -res_max.fun
                    ranges.append(max_val - min_val)
                    bounds_per_dim.append((min_val, max_val))
                else:
                    ranges.append(0)
                    bounds_per_dim.append((None, None))
            
            if not ranges or max(ranges) < 1e-6:
                continue

            split_dim = np.argmax(ranges)
            min_val, max_val = bounds_per_dim[split_dim]
            
            if min_val is not None:
                mid_point = (min_val + max_val) / 2
                
                A1 = np.zeros(self.input_dim); A1[split_dim] = 1.0
                guard1 = node.guard.intersect(Guard(A=np.array([A1]), d=np.array([mid_point])))
                
                A2 = np.zeros(self.input_dim); A2[split_dim] = -1.0
                guard2 = node.guard.intersect(Guard(A=np.array([A2]), d=np.array([-mid_point])))
                
                for new_guard in [guard1, guard2]:
                    if new_guard.is_feasible():
                        _, new_ub = self._get_bounds_in_guard(new_guard)
                        if new_ub > global_lower_bound:
                            heapq.heappush(pq, BBNode(upper_bound=new_ub, guard=new_guard))
        else:
            print(f"B&B reached max iterations ({max_iter}).")
            if pq:
                current_upper_bound = -pq[0].priority
                history.append((max_iter, global_lower_bound, current_upper_bound))

        pbar.close()
        return global_lower_bound, history

# =============================================================================
# --- 公共工具函数 (COMMON UTILITY FUNCTIONS) ---
# =============================================================================

@dataclass
class AuditMetric:
    name: str
    value: Any
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompilationAuditReport:
    metrics: List[AuditMetric]
    failures: List[str]

    def to_dataframe(self) -> pd.DataFrame:
        data = {metric.name: metric.value for metric in self.metrics}
        return pd.DataFrame([data])


def plot_compilation_error(errors, task_name, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(errors, marker='o', linestyle='-', color='r', label='Max Absolute Error')
    plt.axhline(y=np.mean(errors), color='b', linestyle='--', label=f'Average Error: {np.mean(errors):.2e}')
    plt.yscale('log')
    plt.title(f"Per-Sample Dynamic Compilation Error for {task_name}")
    plt.xlabel("Sample Index")
    plt.ylabel("Max Absolute Error (Log Scale)")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(filename)
    print(f"Saved {filename}: Compilation Error plot.")

