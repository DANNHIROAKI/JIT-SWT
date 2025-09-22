import torch
import torch.nn as nn
import numpy as np
import random
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any
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
    """ 定义一个由 Ax <= d 描述的凸多面体区域 """
    A: np.ndarray
    d: np.ndarray

    def intersect(self, other: 'Guard') -> 'Guard':
        if self.A.size == 0: return other
        if other.A.size == 0: return self
        if self.A.shape[1] != other.A.shape[1]:
            raise ValueError("维度不匹配，无法计算守卫的交集")
        new_A = np.vstack([self.A, other.A])
        new_d = np.concatenate([self.d, other.d])
        return Guard(A=new_A, d=new_d)

    def is_feasible(self) -> bool:
        if self.A.size == 0: return True
        c = np.zeros(self.A.shape[1])
        res = linprog(c, A_ub=self.A, b_ub=self.d, bounds=(None, None), method='highs')
        return res.success

@dataclass
class Piece:
    """ 将一个守卫(Guard)与一个仿射变换(weights, bias)绑定 """
    guard: Guard
    weights: np.ndarray
    bias: np.ndarray

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

class ReLuTransition(SymbolicTransition):
    def apply(self, state: CompilationState) -> CompilationState:
        pre_act_flat = state.pre_activation.cpu().numpy().flatten()
        active_map = pre_act_flat > 0
        
        new_guards_A, new_guards_d = [], []
        for i, is_active in enumerate(active_map):
            w_i, b_i = state.piece.weights[i], state.piece.bias[i]
            if is_active:
                new_guards_A.append(-w_i)
                new_guards_d.append(-b_i)
            else:
                new_guards_A.append(w_i)
                new_guards_d.append(b_i)
        
        state.piece.weights[~active_map, :] = 0
        state.piece.bias[~active_map] = 0
        
        if new_guards_A:
            state.piece.guard = state.piece.guard.intersect(Guard(np.array(new_guards_A), np.array(new_guards_d)))
        return state

class FlattenTransition(SymbolicTransition):
    def apply(self, state: CompilationState) -> CompilationState:
        if state.current_shape:
            state.current_shape = (np.prod(state.current_shape),)
        return state

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

class DuanLiu_SWT_Compiler:
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
            self.input_domain = input_domain
        
        self.automaton = WeightedAutomaton(self.model)

    def compile_for_input(self, x_np: np.ndarray) -> Optional[Piece]:
        initial_piece = Piece(
            guard=self.input_domain,
            weights=np.eye(self.input_dim),
            bias=np.zeros(self.input_dim)
        )
        return self.automaton.trace(x_np, initial_piece, self.initial_shape, self.adj_matrix)

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

# =============================================================================
# --- §4.5 Imax分支定界求解器 (Imax Branch & Bound Solver) ---
# =============================================================================

class BranchAndBoundSolver:
    def __init__(self, compiler_F: DuanLiu_SWT_Compiler, compiler_C: DuanLiu_SWT_Compiler, initial_guard: Guard):
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
                res_min = linprog(c, A_ub=node.guard.A, b_ub=node.guard.d, bounds=(None,None))
                res_max = linprog(-c, A_ub=node.guard.A, b_ub=node.guard.d, bounds=(None,None))
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

