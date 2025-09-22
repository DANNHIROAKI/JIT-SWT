import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.datasets import CIFAR10
from torchvision import transforms

from compiler import SWT_Compiler, DEVICE, plot_compilation_error

# =============================================================================
# --- 模型定义 (Model Definition) ---
# =============================================================================
class CNN_A(nn.Module):
    """
    Conv(3->16) -> ReLU -> Conv(16->32) -> ReLU -> GlobalAvgPool -> Linear(32->10)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1) # Global Average Pooling
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)

# =============================================================================
# --- 实验执行函数 (Experiment Execution Function) ---
# =============================================================================

def run():
    print("\n--- Task 2: CNN on CIFAR-10 ---")
    
    def shift_image(x, dx, dy):
        x_s = torch.roll(x, shifts=(dy, dx), dims=(-2, -1))
        if dy > 0: x_s[..., :dy, :] = 0
        elif dy < 0: x_s[..., dy:, :] = 0
        if dx > 0: x_s[..., :, :dx] = 0
        elif dx < 0: x_s[..., :, dx:] = 0
        return x_s

    def check_equivariance_compiled(compiler, dataset, shifts):
        num_samples = 100
        diff_matrix = np.zeros((len(shifts), num_samples))
        indices = np.random.choice(len(dataset), num_samples, replace=False)

        for i, data_idx in enumerate(tqdm(indices, desc="Equivariance Check", leave=False)):
            img_tensor, _ = dataset[data_idx]
            img_np = img_tensor.cpu().numpy()
            
            piece1 = compiler.compile_for_input(img_np)
            if piece1 is None: continue
            out1 = piece1.weights @ img_np.flatten() + piece1.bias
            pred1 = np.argmax(out1)
            
            for j, (dx, dy) in enumerate(shifts):
                shifted_img_tensor = shift_image(img_tensor.unsqueeze(0), dx, dy).squeeze(0)
                shifted_img_np = shifted_img_tensor.cpu().numpy()
                
                piece2 = compiler.compile_for_input(shifted_img_np)
                if piece2 is None: continue
                out2 = piece2.weights @ shifted_img_np.flatten() + piece2.bias
                pred2 = np.argmax(out2)
                
                if pred1 != pred2:
                    diff_matrix[j, i] = 1
        
        num_passed_images = sum(1 for i in range(num_samples) if not diff_matrix[:, i].any())
        pass_rate = num_passed_images / num_samples
        
        diff_counts_per_shift = diff_matrix.sum(axis=1)
        return pass_rate, diff_counts_per_shift.sum(), diff_counts_per_shift

    def plot_cnn_equivariance_heatmap(diff_counts, shifts):
        shift_range = sorted(list(set(s[0] for s in shifts)))
        heatmap_data = np.zeros((len(shift_range), len(shift_range)))
        shift_map = {s: i for i, s in enumerate(shift_range)}

        for count, (dx, dy) in zip(diff_counts, shifts):
            heatmap_data[shift_map[dy], shift_map[dx]] = count
            
        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="viridis",
                          xticklabels=shift_range, yticklabels=shift_range)
        plt.xlabel("Shift in X direction (dx)")
        plt.ylabel("Shift in Y direction (dy)")
        plt.title("Fig 2_1: CNN Equivariance Failure Heatmap")
        plt.savefig("fig2_1_cnn_heatmap.png")
        print("Saved Fig 2_1: CNN Equivariance Failure Heatmap.")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_ds = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_ds = CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    get_subset = lambda d, n: torch.utils.data.Subset(d, np.concatenate([np.where(np.array(d.targets) == i)[0][:n] for i in range(10)]))
    train_subset, test_subset = get_subset(train_ds, 1000), get_subset(test_ds, 200)
    cnn_loader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=2)
    
    model_A = CNN_A().to(DEVICE)
    print("Training CNN Model A...")
    criterion, optimizer = nn.CrossEntropyLoss(), optim.Adam(model_A.parameters(), lr=1e-3)
    
    for epoch in tqdm(range(5), desc="CNN_A Training"):
        model_A.train()
        for inputs, labels in cnn_loader:
            optimizer.zero_grad()
            outputs = model_A(inputs.to(DEVICE))
            loss = criterion(outputs, labels.to(DEVICE))
            loss.backward()
            optimizer.step()
            
    model_A.eval()
    print("CNN_A training complete.")

    compiler_A = SWT_Compiler(model_A, input_dim=3*32*32, initial_shape=(3,32,32))
    
    num_compile_samples = 100
    print(f"Dynamically compiling for {num_compile_samples} CNN samples...")
    errors_cnn = []
    sample_indices_cnn = np.random.choice(len(test_subset), num_compile_samples, replace=False)

    for i in tqdm(sample_indices_cnn, desc="CNN Compilation Test", leave=False):
        test_img_tensor, _ = test_subset[i]
        test_img_np = test_img_tensor.numpy()
        
        local_piece = compiler_A.compile_for_input(test_img_np)
        if local_piece:
            local_eval = local_piece.weights @ test_img_np.flatten() + local_piece.bias
            with torch.no_grad():
                model_eval = model_A(test_img_tensor.unsqueeze(0).to(DEVICE)).cpu().numpy().flatten()
            err = np.max(np.abs(local_eval - model_eval))
            errors_cnn.append(err)

    report = {}
    if errors_cnn:
        plot_compilation_error(errors_cnn, "CNN on CIFAR-10", "fig2_2_cnn_compile_error.png")
        avg_err_cnn = np.mean(errors_cnn)
        report[f"Avg dynamic compile error ({len(errors_cnn)} samples)"] = f"{avg_err_cnn:.2e}"
    else:
        report["Avg dynamic compile error (N/A samples)"] = "Failed"

    shifts = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if not (dx == 0 and dy == 0)]
    pass_A, diff_A_total, diff_counts_per_shift = check_equivariance_compiled(compiler_A, test_subset, shifts)
    report["equivariance pass rate (A, all Δ)"] = f"{pass_A:.3f}"
    report["#diffs linked to boundary"] = diff_A_total
    
    df = pd.DataFrame([report], index=['CNN_A']).T
    print("\nTask 2 CNN Results:")
    print(df)
    
    plot_cnn_equivariance_heatmap(diff_counts_per_shift, shifts)

