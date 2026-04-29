# ai-journey-notebooks
独立完成的人工智能全栈自学项目 | 从Python基础到深度学习前沿技术的完整实践笔记 | 包含CNN/RNN/Transformer/GNN/VAE/GAN等经典架构实现 | 涵盖LoRA/Prompt Tuning等现代微调技术 | 求职作品集
>>>>>>> 6e253eea037d6dfdaa68aa5d39090d83098912d1
# 🔥 AI Self-Learning Journey

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-1.12%2B-red?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Transformers-4.20%2B-orange?logo=huggingface" alt="Transformers">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-yellow?logo=jupyter" alt="Jupyter">
</p>

> **独立完成的人工智能全栈自学项目** | **完整实践笔记** | **求职作品集**

这是一个从零开始、**完全独立完成**的人工智能学习项目，涵盖了从Python基础到深度学习前沿技术的完整学习路径。所有代码、实验和文档均为个人实践成果，旨在系统性掌握AI/ML核心技术栈。

## 🚀 项目亮点

- ✅ **独立完成**：所有内容均为个人自学实践，无任何外部协助
- ✅ **技术全面**：覆盖经典机器学习到现代深度学习的完整技术栈
- ✅ **实战导向**：每个notebook都是可运行的完整实验项目

## ⚠️ 关于文件管理

为了保持仓库的轻量性和快速克隆体验，**仅排除了明确的大型训练数据集和模型权重文件**，具体包括：

- **大型图像数据集**: CNN猫狗数据集 (`猫狗数据集.zip`, `archive.zip`)
- **大型语料库**: WMT中英翻译语料 (`wmt_zh_en_training_corpus.csv`), Europarl语料
- **预训练模型权重**: 所有 `checkpoint-*` 目录下的大型权重文件 (`.bin`, `.pt`, `.safetensors`, `.h5`)
- **大型特征文件**: VGG预提取特征 (`*.tfrecord`), MNIST原始字节文件
- **演示文稿**: 大型PPT文件

### ✅ 保留的重要文件
以下文件**已保留**在仓库中，确保项目的完整性和可复现性：
- **比赛数据**: 长风杯项目的所有SQLite数据库文件
- **小型数据集**: Cora图数据集、MovieLens推荐数据集、Tiny Shakespeare文本
- **文档资料**: 所有PDF论文、技术文档、学习笔记
- **配置文件**: 所有 `.csv`, `.json`, `.txt` 配置和小型数据文件

### 如何获取完整数据？
1. **大型数据集**: 每个notebook中都包含了数据下载和预处理的完整代码
2. **模型训练**: 所有模型都可以通过notebook中的代码重新训练
3. **环境配置**: 只需按照下方指南安装依赖即可运行所有示例

这样设计确保了：
- 🚀 **快速克隆**: 仓库大小控制在合理范围内  
- 💾 **完整保留**: 所有比赛数据和重要文档都在仓库中
- 🔧 **可重现**: 所有结果都可以通过提供的代码重新生成

## 📊 技术栈

### 基础技术
- **编程语言**: Python 3.8+
- **数据处理**: NumPy, Pandas, Scikit-learn
- **可视化**: Matplotlib, Seaborn
- **开发环境**: Jupyter Notebook

### 机器学习
- 线性回归、逻辑回归
- 聚类算法（KMeans, MeanShift）
- 决策树、PCA降维
- 异常检测、模型评估

### 深度学习
- **基础架构**: MLP多层感知器
- **计算机视觉**: CNN卷积神经网络、VGG16、ResNet、批量归一化
- **自然语言处理**: RNN、LSTM、GRU、Seq2Seq、注意力机制
- **前沿架构**: Transformer、GNN图神经网络、VAE变分自编码器、GAN生成对抗网络
- **扩散模型**: 基础扩散模型实现

### 大模型微调
- **BERT微调**: IMDB情感分析 + LoRA/Prompt Tuning
- **GPT-2微调**: Shakespeare文本生成 + 全量微调/LoRA/Prompt Tuning
- **高效微调**: Adapter、LoRA、Prefix Tuning等技术实践
- **Hugging Face生态**: Transformers、Datasets、Accelerate、PEFT

## 📁 目录结构

```
├── 1-python基础与环境配置/           # Python基础和环境配置
├── 2-7-机器学习/                    # 经典机器学习算法系列
│   ├── 2-线性回归/
│   ├── 3-逻辑回归/
│   ├── 4-聚类/
│   ├── 5-常用技术/                 # PCA、决策树、异常检测等
│   └── 6-模型优化与评估/
├── 7-深度学习-MLP多层感知器/        # 深度学习入门
├── 8-深度学习-CNN卷积神经网络/      # 计算机视觉基础
├── 9-12-深度学习-RNN系列/          # 自然语言处理基础
│   ├── 9-RNN循环神经网络/
│   ├── 10-迁移混合学习/
│   ├── 11-ResNet与BN/
│   └── 12-LSTM与GRU/
├── 13-18-深度学习-前沿架构/         # 深度学习进阶
│   ├── 13-Seq2Seq与注意力机制/
│   ├── 14-Transformer/
│   ├── 15-GNN图神经网络/
│   ├── 16-VAE变分自编码器/
│   ├── 17-预训练+微调范式/         # 大模型高效微调
│   └── 18-GAN生成对抗网络-扩散模型/
├── 人工智能笔记/                    # 学习文档和项目指南
└── 长风杯项目/                     # 竞赛项目（包含完整数据集）
```

## 🛠️ 环境配置

### 快速开始
```bash
# 克隆仓库
git clone https://github.com/hzy-sunstone/ai-journey-notebooks.git
cd ai-journey-notebooks

# 创建虚拟环境
python -m venv ai_env
ai_env\Scripts\activate  # Windows
# source ai_env/bin/activate  # Linux/MacOS

# 安装依赖
pip install -r requirements.txt

# 启动Jupyter
jupyter notebook
```

### 核心依赖 (requirements.txt)
```txt
jupyter>=1.0.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
torch>=1.12.0
torchvision>=0.13.0
transformers>=4.20.0
datasets>=2.0.0
accelerate>=0.15.0
peft>=0.4.0
tensorboard>=2.9.0
networkx>=2.8.0
```

## 🎯 学习建议

1. **循序渐进**: 建议按照目录顺序学习，从前到后逐步深入
2. **动手实践**: 每个notebook都包含完整的代码和注释，建议亲自运行调试
3. **重点章节**: 
   - 第17章（大模型微调）展示了现代AI工程的核心技能
   - 第14章（Transformer）是理解现代NLP的基础
   - 第8、11章（CNN、ResNet）是计算机视觉的基石
   - **长风杯项目**: 包含完整的竞赛数据和解决方案


## 📝 关于作者

这是我在人工智能领域的自学成果总结，历时数月完成。通过系统性学习和大量实践，掌握了AI/ML领域的核心技术和最新进展。欢迎关注我的GitHub获取更多技术分享！
个人邮箱：3377431845@qq.com

---

**License**: MIT License  
**Language**: Python 3.8+  
**Framework**: PyTorch, Hugging Face Transformers  
**Last Updated**: 2026年4月