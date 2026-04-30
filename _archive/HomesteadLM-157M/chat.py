"""
MiniMoE-Tiny 交互推理脚本 (BBPE 3270词表版)
用法: python chat.py [--checkpoint <路径>] [--device cpu|cuda] [--temp 0.7]
"""

import torch
import torch.nn.functional as F
import sys, os, time, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.models.minimoe import MiniMoETiny
from tokenizer import Tokenizer  # 使用BBPE分词器


def load_model(checkpoint_path: str, device: str = "cuda"):
    print(f"[*] 加载检查点: {checkpoint_path}")
    print(f"[*] 正在加载权重到 {device}...")
    
    # 加载checkpoint - 使用weights_only=True加速
    ck = torch.load(checkpoint_path, map_location=device, weights_only=True)
    print("[*] 权重加载完成")
    
    config = ck["config"]
    state_dict = ck["model_state_dict"]
    step = ck.get("step", "?")
    
    print(f"[*] 训练步数: {step}")
    print(f"[*] 模型配置: hidden={config['hidden_size']}, layers={config['num_layers']}, "
          f"experts={config['num_experts']}, vocab={config['vocab_size']}")
    
    # 创建模型
    model = MiniMoETiny(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # CUDA优化
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        print(f"[*] CUDA优化: cudnn.benchmark=True")
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[*] 参数量: {total_params:.2f}M")
    
    # 预热推理
    print("[*] 预热推理...")
    dummy_input = torch.randint(0, config["vocab_size"], (1, 10), device=device)
    with torch.no_grad():
        _ = model(dummy_input)
    print("[*] 预热完成，可以开始对话\n")
    
    return model, config


@torch.no_grad()
def generate_stream(
    model,
    input_ids: torch.Tensor,
    config: dict,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    eos_id: int = 2,
    device: str = "cuda",
    verbose: bool = False,
) -> list:
    """流式生成，避免一次性输出"""
    model.eval()
    generated = input_ids.clone()
    prompt_len = input_ids.shape[1]
    
    prev_tokens = set()
    
    for i in range(max_new_tokens):
        # 截断过长的上下文
        if generated.shape[1] > config.get("max_seq_len", 512):
            generated = generated[:, -config.get("max_seq_len", 512):]
        
        # 前向传播
        outputs = model(generated)
        logits = outputs["logits"][:, -1, :].float()
        
        # 重复惩罚
        if repetition_penalty != 1.0:
            for tid in prev_tokens:
                logits[0, tid] /= repetition_penalty
        
        # 温度采样
        if temperature != 1.0:
            logits = logits / temperature
        
        # Top-K
        if top_k > 0:
            kth_val = torch.topk(logits, min(top_k, logits.size(-1)))[0][:, -1, None]
            logits = logits.masked_fill(logits < kth_val, float("-inf"))
        
        # Top-P
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum_probs > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            logits = logits.scatter(1, sorted_idx, sorted_logits.masked_fill(remove, float("-inf")))
        
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # 更新重复惩罚集合
        prev_tokens.add(next_token.item())
        if len(prev_tokens) > 100:
            prev_tokens = set(list(prev_tokens)[-50:])
        
        if next_token.item() == eos_id:
            break
        
        generated = torch.cat([generated, next_token], dim=1)
        
        # 流式输出
        if verbose and i % 20 == 0:
            print(".", end="", flush=True)
    
    return generated[0, prompt_len:].tolist()


def chat_loop(model, tokenizer, config, device: str, args):
    """聊天循环"""
    eos_id = tokenizer.eos_id
    
    print("\n" + "=" * 60)
    print("  MiniMoE-Tiny 对话 (BBPE 3270)")
    print(f"  checkpoint : {os.path.basename(args.checkpoint)}")
    print(f"  device    : {device}")
    print(f"  temp={args.temp}  top_p={args.top_p}  top_k={args.top_k}")
    print()
    print("  命令: /quit 退出 | /clear 清空上下文 | /regen 重新生成")
    print("=" * 60 + "\n")
    
    # 对话历史
    conversation = []
    
    while True:
        try:
            user_text = input("你: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[*] 退出")
            break
        
        if not user_text:
            continue
        
        if user_text.lower() in ("/quit", "/exit", "/q"):
            print("[*] 退出")
            break
        
        if user_text == "/clear":
            conversation = []
            print("[*] 对话历史已清空\n")
            continue
        
        if user_text == "/regen":
            if not conversation:
                print("[!] 没有上一轮回复\n")
                continue
            # 移除最后一条助手回复，重新生成
            conversation = conversation[:-1]
        
        # 添加用户输入
        conversation.append({"role": "user", "content": user_text})
        
        # 构建输入 - 简短指令式
        if len(conversation) == 1:
            # 单轮：直接问
            input_text = f"问: {user_text}\n答:"
        else:
            # 多轮：简短格式
            input_text = ""
            for turn in conversation[-4:]:  # 只保留最近2轮
                if turn["role"] == "user":
                    input_text += f"问: {turn['content']}\n"
                else:
                    input_text += f"答: {turn['content']}\n"
            input_text += "答:"
        
        # 编码
        input_ids = tokenizer.encode(input_text, add_special_tokens=False)
        
        # 生成
        print("AI: ", end="", flush=True)
        t0 = time.time()
        
        out_ids = generate_stream(
            model,
            torch.tensor([input_ids], device=device),
            config,
            max_new_tokens=args.max_tokens,
            temperature=args.temp,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.rep_penalty,
            eos_id=eos_id,
            device=device,
            verbose=True,
        )
        
        elapsed = time.time() - t0
        
        # 解码
        response = tokenizer.decode(out_ids, skip_special_tokens=True)
        
        # 清理BBPE空格
        response = response.replace(" ", "")
        
        print(f"\n{response}")
        print(f"[{len(out_ids)} tokens, {elapsed:.1f}s, {len(out_ids)/elapsed:.0f} tok/s]\n")
        
        # 添加助手回复
        conversation.append({"role": "assistant", "content": response})
        
        # 保持对话长度
        if len(conversation) > 10:
            conversation = conversation[-8:]


def main():
    parser = argparse.ArgumentParser(description="MiniMoE-Tiny 交互推理")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/minimoe_final.pt")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda / cpu")
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--rep_penalty", type=float, default=1.1)
    parser.add_argument("--max_tokens", type=int, default=150)
    args = parser.parse_args()
    
    # 设备
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[!] CUDA不可用，切换到CPU")
        device = "cpu"
    print(f"[*] 使用设备: {device}")
    
    # Tokenizer - 自动使用BBPE
    tokenizer = Tokenizer()
    print(f"[*] 词表: {tokenizer.vocab_size} (BBPE)")
    print(f"[*] 特殊Token: bos={tokenizer.bos_id}, eos={tokenizer.eos_id}, pad={tokenizer.pad_id}")
    
    # 模型路径
    ck_path = args.checkpoint
    if not os.path.isabs(ck_path):
        ck_path = str(Path(__file__).parent / ck_path)
    
    if not os.path.exists(ck_path):
        print(f"[!] 检查点不存在: {ck_path}")
        # 列出可用检查点
        ck_dir = Path(__file__).parent / "checkpoints"
        if ck_dir.exists():
            print(f"[*] 可用检查点:")
            for p in sorted(ck_dir.glob("*.pt")):
                print(f"    {p.name}")
        sys.exit(1)
    
    # 加载模型
    try:
        model, config = load_model(ck_path, device=device)
    except Exception as e:
        print(f"[!] 加载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 开始对话
    chat_loop(model, tokenizer, config, device, args)


if __name__ == "__main__":
    main()
