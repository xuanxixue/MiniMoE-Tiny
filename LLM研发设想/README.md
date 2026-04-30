# HomesteadLM-157M

> 一款面向消费级硬件的轻量级大语言模型 | 157M参数 | MoE架构 | RTX 3050可训练

**项目代号**: MiniMoE-Tiny  
**正式名称**: HomesteadLM-157M  
**作者**: 玄曦雪  
**日期**: 2026-04-16

---

## 📋 简介

HomesteadLM-157M 是一款专为消费级硬件设计的157M参数稀疏门控大语言模型。基于 MoE (Mixture of Experts) 稀疏门控架构，在有限的 GPU 显存（4GB RTX 3050）下成功完成了从零开始的预训练和SFT微调。

## 🎯 核心成果

| 指标 | 值 |
|:-----|:---|
| 总参数量 | 157.10M |
| 活跃参数(每token) | ~13M |
| 预训练步数 | 30,000 |
| 预训练Loss | 0.049 |
| SFT微调步数 | 4,647 |
| SFT Loss | 0.06 |
| 总训练步数 | 34,647 |
| 训练时间 | ~22小时 |
| 词表 | BBPE 3,270 |
| 上下文长度 | 256 |

## 🏗️ 模型架构

```
输入 Token → Embedding → [Transformer Block × 8] → RMSNorm → LM Head → 输出
```

- **注意力**: 分组查询注意力 (GQA) - 8个Q头，4个KV头
- **MoE层**: 8个FFN专家，Top-K=2路由
- **归一化**: RMSNorm + QK-Norm

## 📁 目录结构

```
HomesteadLM-157M/
├── checkpoints/
│   ├── minimoe_final.pt      # 最终模型 (34,647步)
│   └── step_34500.pt         # SFT checkpoint
├── 知识图谱/
│   └── daily_chat_knowledge_graph.json
├── tokenizer/
│   └── tokenizer.json        # BBPE词表 (3,270)
├── config/
├── src/models/
├── 数据集/
│   ├── all_dataset.jsonl     # 预训练数据
│   └── 微调数据集/combined_sft.jsonl  # SFT数据
├── train_streaming.py        # 训练脚本
├── chat.py                   # 对话脚本
├── chat_with_kg.py          # 知识图谱增强对话
└── tokenizer.py
```

## 🚀 快速开始

### 环境要求

```bash
Python 3.12+
PyTorch 2.5.1+ (CUDA 12.1)
```

### 安装依赖

```bash
pip install torch>=2.5.1
```

### 对话演示

```bash
# 使用知识图谱增强对话
python chat_with_kg.py

# 或使用纯模型对话
python chat.py
```

### 从头训练

```bash
python train_streaming.py
```

### 从checkpoint继续训练

```bash
# 自动恢复最新checkpoint
python train_streaming.py --resume

# 指定checkpoint恢复
python train_streaming.py --resume-from step_34500
```

### SFT微调

```bash
# 使用微调数据训练3个epoch
python train_streaming.py --resume --epochs 3
```

## 💬 对话示例

```
你: 你好
AI: 你好！最近怎么样？

你: 今天天气怎么样
AI: [KG] 这么好的天气，有什么出行计划吗？

你: 推荐游戏
AI: [KG] 看你喜欢什么类型的？
```

## 🔧 配置参数

### 模型配置

| 参数 | 值 |
|:-----|:---|
| hidden_size | 512 |
| num_layers | 8 |
| ffn_size | 2048 |
| num_heads | 8 |
| num_kv_heads | 4 |
| num_experts | 8 |
| top_k | 2 |
| vocab_size | 3270 |
| seq_length | 256 |

### 训练配置

| 参数 | 值 |
|:-----|:---|
| batch_size | 4 |
| seq_len | 256 |
| learning_rate | 2e-4 |
| warmup_steps | 100 |
| save_interval | 500 |
| log_interval | 20 |

## 📊 训练曲线

详见 [training_curves.png](LLM研发设想/training_curves.png)

## 📝 论文文档

- [技术路线与研发路线 v2.0](LLM研发设想/MiniMoE-Tiny技术路线与研发路线_v2.0.md)
- [HomesteadLM-157M论文 (英文)](LLM研发设想/HomesteadLM-157M论文.md)
- [HomesteadLM-157M论文 (中文)](LLM研发设想/HomesteadLM-157M论文_中文版.md)

## 🛠️ 工具脚本

| 脚本 | 功能 |
|:-----|:-----|
| train_streaming.py | 流式训练，支持断点续训 |
| chat.py | 基础对话 |
| chat_with_kg.py | 知识图谱增强对话 |
| tokenizer.py | BBPE分词器 |

## 📦 数据集

| 数据集 | 用途 | 规模 |
|:-------|:-----|:-----|
| all_dataset.jsonl | 预训练 | 867.7MB |
| combined_sft.jsonl | SFT微调 | 6,189条QA |

## 🔮 未来计划

- [ ] 扩展上下文到1024+ token
- [ ] 扩展到1B参数规模
- [ ] 知识图谱扩展
- [ ] RAG集成
- [ ] 量化推理优化

## 📜 许可证

MIT License

---

*HomesteadLM-157M - 让大模型训练在消费级硬件上成为可能*
