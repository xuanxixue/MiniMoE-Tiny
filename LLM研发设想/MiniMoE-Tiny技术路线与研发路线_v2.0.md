# HomesteadLM-157M: 低消耗高效AI大语言模型技术文档

**项目代号**: MiniMoE-Tiny  
**正式名称**: HomesteadLM-157M  
**文档版本**: v2.1  
**编制日期**: 2026-04-16  
**目标硬件**: Intel i5-12500H (12代) + 16GB RAM + RTX 3050 4GB  
**训练框架**: Python 3.12 + PyTorch 2.5.1 + CUDA 12.1

---

## 摘要

本文档介绍 **HomesteadLM-157M**，一款专为消费级硬件设计的轻量级大语言模型。基于 MoE (Mixture of Experts) 稀疏门控架构，在 157M 参数规模下实现了高效的训练与推理能力。

**核心成果**：
- 从零训练，总计 **30,000 步**，训练时长 **16.1 小时**
- 最终 Loss: **0.049**
- 在 **RTX 3050 4GB** 显存环境下完成全部训练
- 支持 **BBPE 3270** 词表，中文编码效率高

---

## 第一部分：模型架构

### 1.1 架构概览

```
HomesteadLM-157M 架构：

┌─────────────────────────────────────────────────────────────┐
│                    HomesteadLM-157M                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入 Token Sequence                                         │
│       ↓                                                     │
│  ┌─────────────────┐                                        │
│  │   Token Embed   │  词表: 3,270 (BBPE)                    │
│  └────────┬────────┘                                        │
│           ↓                                                  │
│  ┌─────────────────┐                                        │
│  │   RMSNorm       │  Pre-normalization                      │
│  └────────┬────────┘                                        │
│           ↓                                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │            Transformer Block × 8                        │  │
│  │  ┌─────────────────┐    ┌─────────────────────────┐   │  │
│  │  │   Attention     │    │       MoE Layer         │   │  │
│  │  │  - 8 heads      │    │  - 8 Experts (稀疏87.5%)│   │  │
│  │  │  - GQA (4 KV)   │    │  - Top-K=2 路由          │   │  │
│  │  │  - QK-Norm      │    │  - 辅助损失负载均衡      │   │  │
│  │  └────────┬────────┘    └─────────────┬─────────────┘   │  │
│  │           └──────────┬───────────────┘                 │  │
│  │                      ↓                                  │  │
│  │              ┌─────────────────┐                      │  │
│  │              │    FFN Gate      │  Gate Fusion         │  │
│  │              └────────┬────────┘                      │  │
│  └───────────────────────┼────────────────────────────────┘  │
│                          ↓                                   │
│  ┌─────────────────┐                                        │
│  │     Output      │  LM Head                               │
│  └────────┬────────┘                                        │
│           ↓                                                  │
│  输出 Token 概率分布                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 模型规格

| 参数类别 | HomesteadLM-157M | 说明 |
|:---------|:------------------|:-----|
| **总参数量** | 157.10M | 非Embedding参数 |
| **层数** | 8 | Transformer层 |
| **隐藏维度** | 512 | Hidden size |
| **FFN维度** | 2048 | 前馈网络中间层 |
| **注意力头数** | 8 | Query heads |
| **KV头数** | 4 | Grouped Query Attention |
| **专家数量** | 8 | MoE experts |
| **Top-K** | 2 | 每次激活专家数 |
| **稀疏率** | 87.5% | (8-2)/8 |
| **上下文长度** | 256 | 最大序列长度 |
| **词表大小** | 3,270 | BBPE分词 |
| **激活函数** | GELU | approximate='tanh' |

### 1.3 核心技术组件

#### 1.3.1 MoE稀疏门控

```python
# MoE Layer - 8个FFN专家，Top-K=2
class MoELayer(nn.Module):
    def __init__(self, hidden_size=512, ffn_size=2048, num_experts=8, top_k=2):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, ffn_size),
                nn.GELU(approximate='tanh'),
                nn.Linear(ffn_size, hidden_size),
            ) for _ in range(num_experts)
        ])
        self.top_k = top_k
        self.aux_loss_weight = 0.01

    def forward(self, x):
        B, N, D = x.shape
        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        output = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = top_k_indices[..., k]
            expert_weight = top_k_probs[..., k]
            
            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_output = self.experts[e](x[mask])
                    output[mask] += expert_output * expert_weight[mask].unsqueeze(-1)
        
        # 辅助损失
        gate_probs_mean = gate_probs.mean(dim=[0,1])
        aux_loss = self.num_experts * (gate_probs_mean ** 2).sum()
        
        return output, aux_loss * self.aux_loss_weight
```

#### 1.3.2 注意力机制

```python
# QKNormAttention - Grouped Query Attention
class QKNormAttention(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8, num_kv_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
        # QK-Norm 提升稳定性
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
```

---

## 第二部分：训练配置

### 2.1 训练环境

| 项目 | 配置 |
|:-----|:-----|
| Python | 3.12 |
| PyTorch | 2.5.1+cu121 |
| CUDA | 12.1 |
| GPU | NVIDIA RTX 3050 Laptop (4GB) |
| 训练脚本 | train_streaming.py |
| 优化器 | AdamW |

### 2.2 训练超参数

| 参数 | 值 |
|:-----|:---|
| Batch Size | 4 |
| 序列长度 | 256 |
| 学习率 | 2e-4 |
| Warmup | 100步 |
| 总步数 | 30,000 |
| 保存间隔 | 500步 |
| 日志间隔 | 20步 |

### 2.3 分词器配置

| 项目 | 值 |
|:-----|:---|
| 类型 | BBPE (BPE + ByteLevel) |
| 词表大小 | 3,270 |
| 预分词 | 正则表达式 (GPT-4风格) |
| 特殊Token | bos=1, eos=2, pad=0 |

**BBPE优势**：
- 256字节token + 合并规则
- 中文按字切分，英文按词切分
- 永不OOV（覆盖全部UTF-8字节）

### 2.4 训练数据

| 数据集 | 描述 |
|:-------|:-----|
| all_dataset.jsonl | 867.7MB 流式加载 |
| sft_5000.jsonl | 5,000条高质量问答（微调用） |

---

## 第三部分：训练结果

### 3.1 训练曲线

| 阶段 | Step | Loss | GPU显存 |
|:-----|:-----|:-----|:--------|
| 开始 | 20 | 2.7092 | 2546MB |
| 早期 | 100 | 0.7056 | 2673MB |
| 中期 | 500 | 0.0858 | 2673MB |
| 后期 | 29,500 | 0.0521 | 2672MB |
| 结束 | 30,000 | 0.0492 | 2673MB |

**训练统计**：
- 总耗时: **16.1 小时**
- 生成Token: ~3.1M tokens
- 平均速度: 0.5K tok/s

### 3.2 Loss下降曲线

```
Loss
 3.0 ┤ ●
     │
 2.5 ┤
     │
 2.0 ┤
     │
 1.5 ┤
     │
 1.0 ┤
     │
 0.5 ┤         ●
     │       ●
 0.3 ┤     ●
     │   ●
 0.1 ┤ ●
     └────────────────────────────────── Step
        0    5k   10k   15k   20k   25k   30k
```

### 3.3 推理效果

**测试用例**：
```
问: 你好
答: 你 好 吗？

问: 今天天气
答: 今天 天气 怎么 样？ 助手 : 周末 可以 适当 放松...
```

**观察**：
- 模型能生成多轮对话格式
- BBPE词表导致词与词之间有空格（如"你好"→"你 好"）
- 符合训练数据格式（用户/助手格式）

---

## 第四部分：文件结构

```
MiniMoE-Tiny/ (HomesteadLM-157M)
├── config/
│   └── model_config.py          # 模型配置
├── src/
│   └── models/
│       └── minimoe.py           # 模型实现
├── tokenizer/
│   ├── tokenizer.json          # BBPE分词器
│   ├── vocab.json               # 词表
│   └── tokenizer_config.json    # 分词器配置
├── checkpoints/
│   ├── step_500.pt              # Checkpoint
│   ├── step_1000.pt
│   ├── ...
│   ├── step_30000.pt
│   └── minimoe_final.pt         # 最终模型
├── train_streaming.py           # 训练脚本
├── chat.py                      # 对话脚本
└── tokenizer.py                 # 分词器封装
```

---

## 第五部分：未来规划

### 5.1 已完成 ✅

- [x] 157M参数MoE架构实现
- [x] BBPE 3270词表训练
- [x] 30,000步预训练完成
- [x] RTX 3050 4GB环境适配
- [x] 流式训练管道
- [x] 对话推理脚本
- [x] SFT微调 (6,189条QA数据，3 epochs)
- [x] 知识图谱集成 (72意图、480模板)
- [x] 混合回复系统 (KG + 模型)

### 5.2 待优化 🚧

- [ ] 词表优化（减少BBPE空格）
- [ ] 更大规模预训练
- [ ] 微调数据增强
- [ ] INT8量化推理
- [ ] 评测基准测试

### 5.3 V1-V3 版本架构演进

| 版本 | MoE类型 | CoMeT | 上下文 | 核心特性 |
|:-----|:--------|:------|:-------|:---------|
| **V1 (当前)** | **静态MoE** | ✅ | 256→1024 | 8专家固定路由，Top-K=2 |
| **V2** | **动态MoE** | ✅ | 2048 | 推理时专家新增，门控自适应 |
| **V3** | **动态MoE** | ✅ | 4096+ | ASU膜电位 + Observer门控 |

**关键区分：静态MoE vs 动态MoE**

```
V1 静态MoE ──→ 专家池固定，训练后不变
     ↓
V2/V3 动态MoE ──→ 推理时可动态新增专家，门控在线微调
```

- **CoMeT 是 V1-V3 全版本共有的核心组件**，提供长上下文线性复杂度
- **MoE 在 V1 是静态的**（8专家固定路由），**V2 开始是动态的**（可扩展专家池）

### 5.4 长期目标 🎯

- [ ] 指令遵循能力增强
- [ ] 长上下文支持
- [ ] 多种推理引擎集成
- [ ] 独立exe部署包

---

## 第六部分：技术规格汇总

| 类别 | 规格 |
|:-----|:-----|
| **模型名称** | HomesteadLM-157M |
| **项目代号** | MiniMoE-Tiny |
| **参数量** | 157.10M |
| **架构** | Transformer + MoE |
| **层数** | 8 |
| **隐藏维度** | 512 |
| **专家数** | 8 (Top-K=2) |
| **词表** | BBPE 3,270 |
| **上下文** | 256 |
| **训练步数** | 30,000 |
| **预训练Loss** | 0.049 |
| **SFT步数** | 4,647 |
| **SFT Loss** | 0.06 |
| **总步数** | 34,647 |
| **训练时间** | 16.1h + 5.87h |
| **SFT数据** | 6,189条QA |
| **知识图谱** | 72意图、480模板 |
| **GPU** | RTX 3050 4GB |
| **Python** | 3.12 |
| **PyTorch** | 2.5.1+cu121 |

---

*文档版本：v2.3*  
*最后更新：2026-04-27*  
*项目代号：MiniMoE-Tiny → 正式名称：HomesteadLM-157M*  
*更新内容：明确V1静态MoE，V2/V3动态MoE，CoMeT贯穿全版本*
