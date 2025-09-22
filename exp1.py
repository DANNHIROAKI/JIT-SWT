import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.datasets import MNIST
from torchvision import transforms

from compiler import DuanLiu_SWT_Compiler, DEVICE, plot_compilation_error

# =============================================================================
# --- 模型定义 (Model Definition) ---
# =============================================================================
class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128); self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64); self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

# =============================================================================
# --- 实验执行函数 (Experiment Execution Function) ---
# =============================================================================

def fgsm_attack(model, loss_fn, image_tensor, label_tensor, epsilon):
    image_tensor.requires_grad = True
    output = model(image_tensor)
    loss = loss_fn(output, label_tensor)
    model.zero_grad()
    loss.backward()
    data_grad = image_tensor.grad.data
    perturbed_image = image_tensor + epsilon * data_grad.sign()
    return perturbed_image

def run():
    print("\n--- Task 1: FFN on MNIST - Geometric Analysis & Robustness Correlation ---")
    
    def plot_lipschitz_distribution(l_values):
        plt.figure(figsize=(10, 6))
        sns.histplot(l_values, kde=True, bins=30, color="skyblue")
        plt.title("Fig 1a: Distribution of Local Lipschitz Constants on MNIST Test Set")
        plt.xlabel("Local Lipschitz Constant (L2 norm)")
        plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.75)
        plt.savefig("fig1a_mnist_lipschitz_dist.png")
        print("Saved Fig 1a: Lipschitz Constant Distribution plot.")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = FFN().to(DEVICE)
    criterion, optimizer = nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=1e-3)
    
    print("Training FFN on MNIST...")
    for epoch in tqdm(range(5), desc="FFN MNIST Training"):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.to(DEVICE))
            loss = criterion(outputs, labels.to(DEVICE))
            loss.backward()
            optimizer.step()
            
    model.eval()
    print("FFN training complete.")
    report = {}
    
    compiler = DuanLiu_SWT_Compiler(model, input_dim=28*28, initial_shape=(1, 28, 28))
    print("Dynamically compiling for all test samples to get Lipschitz distribution...")
    
    lipschitz_results = []
    errors = []
    for test_sample_tensor, test_label_tensor in tqdm(test_loader, desc="FFN Full Compilation"):
        test_sample_np = test_sample_tensor.cpu().numpy()
        test_label = test_label_tensor.item()
        
        with torch.no_grad():
            if model(test_sample_tensor.to(DEVICE)).argmax().item() != test_label:
                continue

        local_piece = compiler.compile_for_input(test_sample_np)
        if local_piece:
            with torch.no_grad():
                model_eval = model(test_sample_tensor.to(DEVICE)).cpu().numpy().flatten()
            local_eval = local_piece.weights @ test_sample_np.flatten() + local_piece.bias
            errors.append(np.max(np.abs(local_eval - model_eval)))
            
            l_val = np.linalg.norm(local_piece.weights, ord=2)
            lipschitz_results.append({'l_val': l_val, 'sample': test_sample_tensor, 'label': test_label})
    
    plot_compilation_error(errors, "FFN on MNIST", "fig1b_ffn_compile_error.png")
    report[f"Avg dynamic compile error ({len(errors)} samples)"] = f"{np.mean(errors):.2e}"

    all_l_values = [res['l_val'] for res in lipschitz_results]
    plot_lipschitz_distribution(all_l_values)
    
    lipschitz_results.sort(key=lambda x: x['l_val'], reverse=True)
    N_GROUP = min(50, len(lipschitz_results) // 2)
    high_l_group = lipschitz_results[:N_GROUP]
    low_l_group = lipschitz_results[-N_GROUP:]
    
    print(f"Performing FGSM attack on high/low Lipschitz groups (N={N_GROUP})...")
    epsilon = 0.25
    
    def test_robustness(group, group_name):
        if not group: return 0.0
        successful_attacks = 0
        for item in tqdm(group, desc=f"Attacking {group_name} Group", leave=False):
            sample_tensor = item['sample'].to(DEVICE)
            label_tensor = torch.tensor([item['label']], dtype=torch.long).to(DEVICE)
            perturbed_sample = fgsm_attack(model, criterion, sample_tensor, label_tensor, epsilon)
            with torch.no_grad():
                output = model(perturbed_sample)
                final_pred = output.max(1, keepdim=True)[1]
            if final_pred.item() != item['label']:
                successful_attacks += 1
        return successful_attacks / len(group)

    high_l_success_rate = test_robustness(high_l_group, "High-L")
    low_l_success_rate = test_robustness(low_l_group, "Low-L")
    
    L_upper = compiler.get_lipschitz_upper_bound()
    report["Global Lipschitz Upper Bound (L_upper)"] = f"{L_upper:.4f}"
    report[f"Avg Local-L (Top {N_GROUP})"] = f"{np.mean([g['l_val'] for g in high_l_group]):.4f}"
    report[f"Avg Local-L (Bottom {N_GROUP})"] = f"{np.mean([g['l_val'] for g in low_l_group]):.4f}"
    report[f"Attack Success Rate (Top {N_GROUP}, eps={epsilon})"] = f"{high_l_success_rate:.2%}"
    report[f"Attack Success Rate (Bottom {N_GROUP}, eps={epsilon})"] = f"{low_l_success_rate:.2%}"

    print("\n--- Task 1 FFN & Robustness Results ---")
    print(pd.DataFrame([report], index=['value']).T)

