import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import networkx as nx
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
from matplotlib.patches import Patch

from compiler import (
    SWT_Compiler, BranchAndBoundSolver, Guard, DEVICE
)


# =============================================================================
# --- 模型定义 (Model Definitions) ---
# =============================================================================
class GCNConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCNConv, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, adj):
        support = torch.spmm(adj, x)
        output = self.linear(support)
        return output


class GNN(nn.Module):
    def __init__(self, in_dim=6, hidden_dim=16, out_dim=2):
        super(GNN, self).__init__()
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, adj):
        h1 = self.relu1(self.gcn1(x, adj))
        h2 = self.relu2(self.gcn2(h1, adj))
        return self.fc(h2)


class Intervention(nn.Module):
    # ==========================================================================
    # --- MODIFICATION START ---
    # 修复了 AttributeError。
    # 我们为 Intervention 模块添加了 .channel 属性，以兼容不可修改的 compiler.py。
    # 这个 .channel 属性仅在初始化时传入的通道列表长度为1时才被创建。
    # 这恰好满足了编译器的需求，因为它只在计算单个通道的 Imax 时被调用。
    # 而 self.channels 属性则继续用于支持在 forward pass 中对多个通道进行干预。
    # ==========================================================================
    def __init__(self, channels: List[int], num_nodes: int, hidden_dim: int):
        super().__init__()
        self.channels = channels
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # HACK: 为兼容旧版 compiler.py (该文件只支持单个通道且不可修改),
        # 添加 .channel 属性。
        # 这依赖于一个事实：只有在为单个通道计算Imax时，即 len(channels) == 1 时，
        # 该模块实例才会被传入编译器。
        if len(channels) == 1:
            self.channel = channels[0]

    def forward(self, h):
        h_c = h.clone()
        if h_c.dim() == 2 and h_c.shape[0] == self.num_nodes and h_c.shape[1] == self.hidden_dim:
            h_c[:, self.channels] = 0
        return h_c
    # --- MODIFICATION END ---
    # ==========================================================================


class IntervenedGNN(nn.Module):
    """
    一个包装模型，它包含原始GNN的组件和一个干预层。
    这个模型可以正确处理GNN的(features, adj)双输入。
    """

    def __init__(self, original_model: GNN, channels_to_intervene: List[int], num_nodes: int, hidden_dim: int):
        super().__init__()
        self.gcn1 = original_model.gcn1
        self.relu1 = original_model.relu1
        self.gcn2 = original_model.gcn2
        self.relu2 = original_model.relu2
        self.intervention = Intervention(channels_to_intervene, num_nodes, hidden_dim)
        self.fc = original_model.fc

    def forward(self, x, adj):
        h1 = self.relu1(self.gcn1(x, adj))
        h2 = self.relu2(self.gcn2(h1, adj))
        h_intervened = self.intervention(h2)
        return self.fc(h_intervened)


# =============================================================================
# --- 扩展实验辅助函数 (Helper Functions for Extended Experiment) ---
# =============================================================================

def verify_compilation_equivalence(
        model: GNN,
        compiler: SWT_Compiler,
        features_tensor: torch.Tensor,
        adj_tensor: torch.Tensor,
        num_tests: int = 100
) -> List[float]:
    """验证符号编译器和原始PyTorch模型的输出是否一致 (Forward Equivalence)"""
    errors = []
    model.eval()
    print(f"Running compilation equivalence check for {num_tests} samples...")
    for i in tqdm(range(num_tests), desc="Verifying Compilation"):
        # 对输入添加微小扰动以测试不同点
        noise = torch.randn_like(features_tensor) * 0.01
        test_features = features_tensor + noise

        # 1. 获取原始模型输出
        with torch.no_grad():
            y_model = model(test_features, adj_tensor).cpu().numpy()

        # 2. 获取编译器输出
        x_np = test_features.cpu().numpy()
        piece = compiler.compile_for_input(x_np)

        if piece is None:
            print(f"Warning: Compilation failed for sample {i}.")
            errors.append(float('inf'))
            continue

        x_flat = x_np.flatten()
        y_compiled_flat = piece.weights @ x_flat + piece.bias
        y_compiled = y_compiled_flat.reshape(y_model.shape)

        # 3. 计算并记录误差
        error = np.max(np.abs(y_model - y_compiled))
        errors.append(error)

    return errors


def verify_permutation_equivariance(
        model: GNN,
        features: np.ndarray,
        adj: np.ndarray,
        num_nodes: int,
        num_permutations: int = 50
) -> List[float]:
    """使用符号编译器验证GNN模型的置换等变性"""
    errors = []
    model.eval()
    print(f"Running permutation equivariance check with compiler for {num_permutations} permutations...")

    # --- 1. 计算原始输入的符号化输出 F(x) ---
    adj_loop_orig = adj + np.eye(num_nodes)
    d_inv_sqrt_orig = np.power(adj_loop_orig.sum(axis=1), -0.5).flatten()
    d_inv_sqrt_orig[np.isinf(d_inv_sqrt_orig)] = 0.
    norm_adj_orig = np.diag(d_inv_sqrt_orig).dot(adj_loop_orig).dot(np.diag(d_inv_sqrt_orig))
    adj_tensor_orig = torch.tensor(norm_adj_orig, dtype=torch.float32).to(DEVICE)

    compiler_orig = SWT_Compiler(
        model,
        input_dim=features.size,
        initial_shape=features.shape,
        adj_matrix=adj_tensor_orig
    )

    piece_orig = compiler_orig.compile_for_input(features)
    if piece_orig is None:
        raise RuntimeError("Compiler failed for the original input during equivariance check.")

    out_dim = model.fc.out_features
    y_orig_flat = piece_orig.weights @ features.flatten() + piece_orig.bias
    y_orig = y_orig_flat.reshape((num_nodes, out_dim))

    for _ in tqdm(range(num_permutations), desc="Verifying Equivariance with Compiler"):
        pi = np.random.permutation(num_nodes)
        P = np.eye(num_nodes)[pi]

        # --- 2. 置换输入 Px ---
        features_perm = P @ features
        adj_perm = P @ adj @ P.T

        # 重新归一化置换后的邻接矩阵
        adj_loop_perm = adj_perm + np.eye(num_nodes)
        d_inv_sqrt_perm = np.power(adj_loop_perm.sum(axis=1), -0.5).flatten()
        d_inv_sqrt_perm[np.isinf(d_inv_sqrt_perm)] = 0.
        norm_adj_perm = np.diag(d_inv_sqrt_perm).dot(adj_loop_perm).dot(np.diag(d_inv_sqrt_perm))
        adj_tensor_perm = torch.tensor(norm_adj_perm, dtype=torch.float32).to(DEVICE)

        # --- 3. 计算置换输入的符号化输出 F(Px) ---
        compiler_perm = SWT_Compiler(
            model,
            input_dim=features_perm.size,
            initial_shape=features_perm.shape,
            adj_matrix=adj_tensor_perm
        )
        piece_perm = compiler_perm.compile_for_input(features_perm)

        if piece_perm is None:
            print(f"Warning: Compilation failed for a permuted sample.")
            errors.append(float('inf'))
            continue

        y_perm_flat = piece_perm.weights @ features_perm.flatten() + piece_perm.bias
        y_perm = y_perm_flat.reshape((num_nodes, out_dim))

        # --- 4. 期望的置换输出 P(F(x)) ---
        expected_output_perm = P @ y_orig

        # --- 5. 计算误差 ---
        error = np.max(np.abs(y_perm - expected_output_perm))
        errors.append(error)

    return errors


def calculate_imax_for_channel(
        model: GNN,
        compiler_F: SWT_Compiler,
        initial_guard: Guard,
        channel_to_intervene: int,
        num_nodes: int,
        hidden_dim: int,
        features_tensor: torch.Tensor,
        adj_tensor: torch.Tensor,
        max_iter: int = 50
) -> Tuple[float, List]:
    # 创建被干预的模型时，将单个通道包装在列表中
    model_C = IntervenedGNN(
        model, [channel_to_intervene], num_nodes, hidden_dim
    ).to(DEVICE).eval()

    compiler_C = SWT_Compiler(
        model_C,
        input_dim=features_tensor.numel(),
        initial_shape=features_tensor.shape,
        adj_matrix=adj_tensor
    )

    solver = BranchAndBoundSolver(compiler_F, compiler_C, initial_guard)
    imax_value, imax_history = solver.solve(max_iter=max_iter, tolerance=1e-2)
    return imax_value, imax_history


def evaluate_accuracy(model: nn.Module, features: torch.Tensor, adj: torch.Tensor, labels: torch.Tensor) -> float:
    """评估模型在给定数据上的准确率"""
    model.eval()
    with torch.no_grad():
        outputs = model(features, adj)
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == labels).sum().item()
        return correct / len(labels)


# =============================================================================
# --- 绘图函数 (Plotting Functions) ---
# =============================================================================

def plot_compilation_verification_errors(errors: List[float], filename: str):
    """绘制编译验证误差的分布图"""
    plt.figure(figsize=(10, 6))
    plt.plot(errors, marker='o', linestyle='-', color='r', label='Max Absolute Error')
    plt.axhline(y=np.mean(errors), color='b', linestyle='--', label=f'Average Error: {np.mean(errors):.2e}')
    plt.yscale('log')
    plt.title(f"Fig 1: Compilation Equivalence Verification Error")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Max Absolute Error (Log Scale)")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}: Compilation verification plot.")


def plot_equivariance_verification_errors(errors: List[float], filename: str):
    """绘制置换等变性验证误差的分布图"""
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=20, color='purple', alpha=0.7, label='Error Distribution')
    plt.axvline(x=np.mean(errors), color='r', linestyle='--', label=f'Average Error: {np.mean(errors):.2e}')
    plt.yscale('log')
    plt.title("Fig 2: GNN Permutation Equivariance Verification Error (using Compiler)")
    plt.xlabel("Max Absolute Error (Log Scale)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}: Equivariance verification plot.")


def plot_imax_convergence(history: List, channel_id: int, filename: str):
    """ 绘制Imax分支定界算法的收敛过程 """
    if not history:
        print(f"Warning: Imax history for channel {channel_id} is empty, cannot generate convergence plot.")
        return

    iterations = [h[0] for h in history]
    lower_bounds = [h[1] for h in history]
    upper_bounds = [h[2] for h in history]

    plt.figure(figsize=(12, 7))
    plt.plot(iterations, lower_bounds, 'o-', label='Global Lower Bound', color='blue', markersize=4)
    plt.plot(iterations, upper_bounds, 'o-', label='Global Upper Bound', color='red', markersize=4)
    plt.fill_between(iterations, lower_bounds, upper_bounds, color='gray', alpha=0.2, label='Uncertainty Gap')

    final_imax = lower_bounds[-1]
    plt.axhline(y=final_imax, color='green', linestyle='--', label=f'Final Imax Value: {final_imax:.4f}')

    plt.title(f"Fig 3: Channel {channel_id} - Imax B&B Algorithm Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Imax Value")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}: Imax convergence plot for channel {channel_id}.")


def plot_imax_distribution(all_imax_values: List[Tuple[int, float]], filename: str):
    # 更新绘图逻辑，以高亮显示Imax最高和最低的5个通道
    all_imax_values_sorted_by_channel = sorted(all_imax_values, key=lambda x: x[0])
    channels = [item[0] for item in all_imax_values_sorted_by_channel]
    imax_vals = [item[1] for item in all_imax_values_sorted_by_channel]

    all_imax_values_sorted_by_val = sorted(all_imax_values, key=lambda x: x[1])
    bottom_5_channels = {item[0] for item in all_imax_values_sorted_by_val[:5]}
    top_5_channels = {item[0] for item in all_imax_values_sorted_by_val[-5:]}

    colors = []
    for ch in channels:
        if ch in top_5_channels:
            colors.append('salmon')
        elif ch in bottom_5_channels:
            colors.append('lightgreen')
        else:
            colors.append('skyblue')

    plt.figure(figsize=(14, 7))
    plt.bar(channels, imax_vals, color=colors)
    plt.xlabel("Channel Index")
    plt.ylabel("Theoretical Imax")
    plt.title("Fig 4: Imax Influence Distribution Across GNN Hidden Channels")
    plt.xticks(channels)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    legend_elements = [
        Patch(facecolor='salmon', label='Top-5 Imax Channels'),
        Patch(facecolor='lightgreen', label='Bottom-5 Imax Channels'),
        Patch(facecolor='skyblue', label='Other Channels')
    ]
    plt.legend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}: Imax distribution plot.")


def plot_accuracy_impact(
        baseline_acc: float,
        high_impact_acc: float,
        low_impact_acc: float,
        filename: str
):
    # 更新绘图标签以反映分组对比
    """可视化干预对模型准确率的影响"""
    labels = ['Baseline Model', 'Remove Top-5 Imax Channels', 'Remove Bottom-5 Imax Channels']
    accuracies = [baseline_acc, high_impact_acc, low_impact_acc]

    plt.figure(figsize=(10, 7))
    bars = plt.bar(labels, accuracies, color=['royalblue', 'salmon', 'lightgreen'])
    plt.ylabel("Model Prediction Accuracy")
    plt.title("Fig 5: Performance Impact of Removing Channel Groups")
    plt.ylim(0, 1.05)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.01, f'{yval:.2%}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}: Accuracy impact plot.")


# =============================================================================
# --- 实验主函数 (Main Experiment Function) ---
# =============================================================================

def run():
    print("\n--- Task 3 (Enhanced): GNN Compilation, Property Verification, and Imax Analysis ---")
    G = nx.karate_club_graph()
    num_nodes = G.number_of_nodes()

    labels = np.array([1 if G.nodes[i]['club'] == 'Officer' else 0 for i in sorted(G.nodes())], dtype=np.int64)
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(DEVICE)

    # 保存原始邻接矩阵用于等变性测试
    adj = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))

    adj_loop = adj + np.eye(num_nodes)
    d_inv_sqrt = np.power(adj_loop.sum(axis=1), -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    norm_adj = np.diag(d_inv_sqrt).dot(adj_loop).dot(np.diag(d_inv_sqrt))
    adj_tensor = torch.tensor(norm_adj, dtype=torch.float32).to(DEVICE)

    features = StandardScaler().fit_transform(np.hstack([
        np.array([d for _, d in G.degree()])[:, None],
        np.array(list(nx.clustering(G).values()))[:, None],
        eigh(nx.laplacian_matrix(G).astype(float).toarray())[1][:, 1:5]
    ]))
    features_tensor = torch.tensor(features, dtype=torch.float32).to(DEVICE)

    hidden_dim_gnn = 16
    model = GNN(in_dim=features_tensor.shape[1], hidden_dim=hidden_dim_gnn, out_dim=2).to(DEVICE)
    criterion, optimizer = nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=1e-2)

    print("Training GNN...")
    for epoch in tqdm(range(200), desc="GNN Training"):
        model.train();
        optimizer.zero_grad()
        outputs = model(features_tensor, adj_tensor)
        loss = criterion(outputs, labels_tensor);
        loss.backward();
        optimizer.step()
    model.eval();
    print("GNN training complete.")

    compiler_F = SWT_Compiler(model, input_dim=features_tensor.numel(), initial_shape=features_tensor.shape,
                                      adj_matrix=adj_tensor)

    # --- 新增步骤 1: 编译验证 ---
    print("\n--- Step 1: Compilation Equivalence Verification ---")
    comp_errors = verify_compilation_equivalence(model, compiler_F, features_tensor, adj_tensor)
    plot_compilation_verification_errors(comp_errors, "fig3_1_compilation_verification.png")

    # --- 新增步骤 2: GNN 性质验证 (置换等变性) ---
    print("\n--- Step 2: GNN Property Verification (Permutation Equivariance) ---")
    equiv_errors = verify_permutation_equivariance(model, features, adj, num_nodes)
    plot_equivariance_verification_errors(equiv_errors, "fig3_2_equivariance_verification.png")

    # --- 步骤 3: 计算所有隐藏通道的 Imax ---
    print("\n--- Step 3: Calculating Imax for all hidden channels ---")
    all_imax_values = []
    eps = 0.1
    input_dim_flat = features_tensor.numel()
    x0_flat = features_tensor.cpu().numpy().flatten()
    A_domain = np.vstack([np.eye(input_dim_flat), -np.eye(input_dim_flat)])
    d_domain = np.concatenate([x0_flat + eps, -(x0_flat - eps)])
    initial_guard = Guard(A=A_domain, d=d_domain)

    for channel in tqdm(range(hidden_dim_gnn), desc="Calculating Imax per channel"):
        # 为了演示，我们只对第一个通道进行完整的B&B搜索
        max_iter = 100 if channel == 0 else 10
        imax_val, imax_history = calculate_imax_for_channel(
            model, compiler_F, initial_guard, channel,
            num_nodes, hidden_dim_gnn, features_tensor, adj_tensor, max_iter=max_iter
        )
        if channel == 0:
            plot_imax_convergence(imax_history, channel, "fig3_3_convergence_channel_0.png")
        all_imax_values.append((channel, imax_val))

    # --- 步骤 4: 识别关键通道组并绘制分布图 ---
    print("\n--- Step 4: Identifying key channel groups and plotting distributions ---")
    all_imax_values.sort(key=lambda x: x[1])  # 按Imax值从小到大排序

    # 因为隐藏层维度变小，这里调整为Top-2和Bottom-2
    channels_low_imax = [item[0] for item in all_imax_values[:5]]
    channels_high_imax = [item[0] for item in all_imax_values[-5:]]

    plot_imax_distribution(all_imax_values, "fig3_4_imax_distribution.png")

    # --- 步骤 5: 性能影响验证 ---
    print("\n--- Step 5: Performance Impact Validation ---")

    baseline_accuracy = evaluate_accuracy(model, features_tensor, adj_tensor, labels_tensor)

    print(f"Removing Top-5 Imax channels: {sorted(channels_high_imax)}")
    model_C_high = IntervenedGNN(model, channels_high_imax, num_nodes, hidden_dim_gnn).to(DEVICE).eval()
    accuracy_after_high_impact = evaluate_accuracy(model_C_high, features_tensor, adj_tensor, labels_tensor)

    print(f"Removing Bottom-5 Imax channels: {sorted(channels_low_imax)}")
    model_C_low = IntervenedGNN(model, channels_low_imax, num_nodes, hidden_dim_gnn).to(DEVICE).eval()
    accuracy_after_low_impact = evaluate_accuracy(model_C_low, features_tensor, adj_tensor, labels_tensor)

    # --- 步骤 6: 分析与最终报告 ---
    print("\n--- Step 6: Analysis and final report ---")
    plot_accuracy_impact(
        baseline_accuracy,
        accuracy_after_high_impact,
        accuracy_after_low_impact,
        "fig3_5_accuracy_impact.png"
    )

    report_data = {
        "Verification Results": "---",
        "Max Compilation Error": f"{np.max(comp_errors):.2e}",
        "Mean Compilation Error": f"{np.mean(comp_errors):.2e}",
        "Max Equivariance Error": f"{np.max(equiv_errors):.2e}",
        "Mean Equivariance Error": f"{np.mean(equiv_errors):.2e}",
        "Imax Analysis Results": "---",
        "Baseline Accuracy": f"{baseline_accuracy:.2%}",
        "Top-5 High Imax Channels": str(sorted(channels_high_imax)),
        "Accuracy after removing Top-5": f"{accuracy_after_high_impact:.2%}",
        "Performance Drop (Top-5)": f"{baseline_accuracy - accuracy_after_high_impact:.2%}",
        "Bottom-5 Low Imax Channels": str(sorted(channels_low_imax)),
        "Accuracy after removing Bottom-5": f"{accuracy_after_low_impact:.2%}",
        "Performance Drop (Bottom-5)": f"{baseline_accuracy - accuracy_after_low_impact:.2%}",
    }

    report_df = pd.DataFrame(list(report_data.items()), columns=['Metric', 'Value'])

    print("\n--- Task 3 Final GNN Results ---")
    print(report_df.to_string(index=False))


if __name__ == "__main__":
    run()


