# JIT-SWT: Just-In-Time Piecewise-Linear Semantics for ReLU-type Networks

本项目是论文《JUST-IN-TIME PIECEWISE-LINEAR SEMANTICS FOR RELU-TYPE NETWORKS》的官方代码实现。

## 摘要

本项目提出了一种新颖的、用于分析含ReLU类激活函数的神经网络的方法。传统分析方法通常会面临由分段线性(Piecewise-Linear)区域数量指数级增长带来的“表达式爆炸”问题。为解决此难题，我们引入了**符号加权转换器 (Symbolic Weighted Transducer, SWT)** 作为网络的形式化表示，并进一步提出了**即时编译 (Just-In-Time, JIT)** 语义。该方法避免了对网络进行全局的、静态的展开，而是根据给定的输入或输入域，进行按需的、局部的分析。JIT编译器只在必要时才添加超平面（即“守卫”，Guards），从而将对全连接网络（FFN）、卷积网络（CNN）和图神经网络（GNN）等模型的几何分析与形式验证变得 tractable。

## 核心概念

- **符号加权转换器 (SWT)**: 我们将神经网络编译成一种形式化的自动机。在这个自动机中，状态间的转移由多面体区域（Guards）守护，并携带一个仿射变换（affine transformation）。网络的复杂行为被精确地映射为在这个带权自动机上的路径计算。
- **即时编译 (JIT)**: 这是本框架的核心优势。相较于传统方法试图枚举网络所有的线性区域，JIT编译器在接收到一个具体输入（或输入域）时，才“即时”地、局部地推导出网络在该点附近的分段线性语义。这种“按需精化”（on-demand refinement）的策略，使得我们能够高效地获得精确的局部线性表达式（f(x′)=Wx′+b），从而绕开了全局分析的指数级复杂度。

## 文件结构

- `compiler.py`: 框架的核心实现。包含了 JIT-SWT 编译器的主要逻辑，以及用于计算最大因果影响（Imax）等性质的分支定界（Branch and Bound）求解器。
- `exp1.py`: **实验一**: FFN @ MNIST - 局部利普希茨常数 vs. 对抗鲁棒性。
- `exp2.py`: **实验二**: CNN @ CIFAR-10 - 平移等变性 (Translation Equivariance) 验证。
- `exp3.py`: **实验三**: GNN @ Karate Club - 置换等变性 (Permutation Equivariance) 与因果影响分析。
- `run.py`: 用于复现所有实验的主入口脚本。
- `ICLR.pdf`: 本项目的研究论文。

## 安装

1. 克隆本仓库:

   ```
   git clone https://github.com/Rye-wisky/SWT
   cd SWT
   ```

2. 安装依赖环境:

   ```
   pip install -r requirements.txt
   ```
   
## 复现实验

您可以通过 `run.py` 脚本轻松复现论文中的所有实验。

### **实验一: FFN 局部几何分析与鲁棒性**

- **运行命令**:

  ```
  python run.py --exp 1
  ```

- **实验目的**: 验证JIT编译器能够精确提取FFN在某输入点的局部仿射表达，并探究局部利普希茨常数与模型对抗鲁棒性之间的关联。

- **预期输出**:

  1. 控制台将打印出编译误差、全局利普希茨上界、高/低敏感度样本组的平均局部利普希茨常数，以及FGSM攻击成功率的对比表格。
  2. 项目目录下将生成两张图表：
     - `fig1a_mnist_lipschitz_dist.png`: MNIST测试集上局部利普希茨常数的分布直方图。
     - `fig1b_ffn_compile_error.png`: FFN的逐样本动态编译误差图。

### **实验二: CNN 平移等变性验证**

- **运行命令**:

  ```
  python run.py --exp 2
  ```

- **实验目的**: 使用JIT编译器验证一个小型CNN模型在输入发生微小平移时的平移等变性，并定位导致性质破坏的原因。

- **预期输出**:

  1. 控制台将打印编译误差以及在不同平移下的等变性通过率。
  2. 项目目录下将生成两张图表：
     - `fig2_1_cnn_heatmap.png`: 等变性验证失败次数在不同平移方向上的热力图。
     - `fig2_2_cnn_compile_error.png`: CNN的逐样本动态编译误差图。

### **实验三: GNN 性质验证与因果分析**

- **运行命令**:

  ```
  python run.py --exp 3
  ```

- **实验目的**: 全面展示JIT-SWT框架在GNN上的高级分析能力，包括编译等价性验证、置换等变性验证，以及识别关键神经元（通道）的因果推断。

- **预期输出**:

  1. 控制台将打印一份详细的报告，包括编译与等变性验证的误差、基线模型准确率，以及移除高/低影响力通道后的性能对比。
  2. 项目目录下将生成五张图表：
     - `fig3_1_compilation_verification.png`: GNN编译等价性验证误差图。
     - `fig3_2_equivariance_verification.png`: GNN置换等变性验证误差图。
     - `fig3_3_convergence_channel_0.png`: 对通道0计算Imax时，分支定界算法的收敛过程。
     - `fig3_4_imax_distribution.png`: GNN隐藏层所有通道的Imax影响力分布图。
     - `fig3_5_accuracy_impact.png`: 移除高/低Imax通道组后对模型准确率影响的对比柱状图。

## 实验详解与结论

#### 实验一: FFN @ MNIST

本实验训练了一个简单的全连接网络。JIT编译器被用于计算每个测试样本点对应的局部仿射函数 f(x′)=Wx′+b。该函数的雅可比矩阵 W 的谱范数 ∣∣W∣∣_2 即为该点的局部利普希茨常数，它衡量了模型输出对输入的局部敏感度。

结论: 实验结果清晰地表明，局部利普希茨常数与对抗攻击成功率呈强正相关。局部敏感度最高的Top-50样本在FGSM攻击下的成功率为100%，而敏感度最低的Bottom-50样本仅为14%。这证明了我们的框架可以有效识别模型中的脆弱点。

#### 实验二: CNN @ CIFAR-10

本实验训练了一个小型CNN。编译器被用于验证其平移等变性——即当输入图像发生平移时，输出也应相应平移。

结论: 模型在8种不同方向的平移下总体通过率为85%。通过分析失败案例的热力图，我们发现失败主要集中在对角线方向的平移上。这与理论预期一致，证明了等变性的破坏主要源于零填充（Zero Padding） 引入的边界效应。

#### 实验三: GNN @ Karate Club

这是最全面的一个实验，在一个GNN上展示了JIT-SWT框架的多种高级应用：

1. **编译与性质验证**: 验证了编译器输出与原Pytorch模型输出在数值上的一致性（误差约为 10−6），并以极高的精度验证了GNN的**置换等变性**（Permutation Equivariance），这是GNN的核心性质。
2. **最大因果影响 (Imax)**: 我们提出并使用内置的分支定界求解器计算了`Imax`。`Imax` 精确地量化了对GNN中某个隐藏通道进行干预（例如，将其输出置零）对模型最终输出可能造成的最大影响。
3. **Imax指导的神经元裁剪**: 通过比较所有通道的`Imax`值，我们识别出了最重要和最不重要的通道。消融实验证实：移除`Imax`最高的Top-5通道，模型准确率显著下降**2.94%**；而移除`Imax`最低的Bottom-5通道，准确率**不受影响**。

**结论**: `Imax`是一个非常有效的衡量标准，可以用于定位和理解神经网络中的关键计算单元，为模型压缩、剪枝和可解释性分析提供了坚实的理论与工具支持。

## 引用

如果您在您的研究中使用了本代码或思想，请引用我们的论文：

```
@article{anonymous2026jit,
  title={JUST-IN-TIME PIECEWISE-LINEAR SEMANTICS FOR RELU-TYPE NETWORKS},
  author={Anonymous},
  journal={Submitted to International Conference on Learning Representations},
  year={2026},
  url={[https://openreview.net/forum?id=](https://openreview.net/forum?id=)...}
}
```