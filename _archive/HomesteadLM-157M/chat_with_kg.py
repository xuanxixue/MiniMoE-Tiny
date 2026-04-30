"""
HomesteadLM-157M 知识图谱增强对话系统
结合MiniMoE模型推理 + 知识图谱模板匹配
"""

import torch
import torch.nn.functional as F
import sys, os, time, argparse, json, re, random
from pathlib import Path
from difflib import SequenceMatcher

sys.path.insert(0, str(Path(__file__).parent))

from src.models.minimoe import MiniMoETiny
from tokenizer import Tokenizer


class KnowledgeGraph:
    """知识图谱问答系统"""
    
    def __init__(self, kg_path: str):
        print(f"[*] 加载知识图谱: {kg_path}")
        with open(kg_path, 'r', encoding='utf-8') as f:
            self.kg = json.load(f)
        
        self.intent_patterns = {}  # intent -> [(pattern, weight)]
        self.intent_templates = {}  # intent -> [templates]
        self.intent_keywords = {}  # intent -> [keywords]
        
        self._build_index()
        
        meta = self.kg.get("meta", {})
        print(f"[*] 知识图谱: {meta.get('total_intents', 0)}意图, "
              f"{meta.get('total_entities', 0)}实体, "
              f"{meta.get('total_templates', 0)}模板")
    
    def _build_index(self):
        """构建快速查询索引"""
        for topic_name, topic_data in self.kg.get("topics", {}).items():
            for subtopic_name, subtopic in topic_data.get("subtopics", {}).items():
                intent = subtopic.get("intent", f"{topic_name}.{subtopic_name}")
                
                # 收集patterns
                patterns = subtopic.get("patterns", [])
                self.intent_patterns[intent] = [(p, 1.0) for p in patterns]
                
                # 收集keywords
                keywords = subtopic.get("keywords", [])
                self.intent_keywords[intent] = keywords
                
                # 收集templates
                templates = []
                for rt in subtopic.get("response_templates", []):
                    templates.extend(rt.get("templates", []))
                self.intent_templates[intent] = templates
    
    def match(self, user_input: str) -> tuple:
        """
        匹配用户输入
        返回: (intent, response, confidence)
        """
        text = user_input.strip()
        
        best_match = None
        best_score = 0.0
        best_intent = None
        
        # 1. 精确pattern匹配
        for intent, patterns in self.intent_patterns.items():
            for pattern, weight in patterns:
                if re.search(pattern, text):
                    score = weight * 1.0
                    if score > best_score:
                        best_score = score
                        best_intent = intent
        
        # 2. 关键词匹配
        if not best_match or best_score < 0.8:
            for intent, keywords in self.intent_keywords.items():
                matched = sum(1 for kw in keywords if kw in text)
                if matched > 0:
                    score = matched / len(keywords) * 0.8
                    if score > best_score:
                        best_score = score
                        best_intent = intent
        
        # 3. 模糊匹配keywords
        if not best_match or best_score < 0.5:
            for intent, keywords in self.intent_keywords.items():
                for kw in keywords:
                    ratio = SequenceMatcher(None, text, kw).ratio()
                    if ratio > 0.8:
                        score = ratio * 0.6
                        if score > best_score:
                            best_score = score
                            best_intent = intent
        
        # 生成回复
        if best_intent and self.intent_templates.get(best_intent):
            response = random.choice(self.intent_templates[best_intent])
            return best_intent, response, min(best_score, 1.0)
        
        return None, None, 0.0


class KGEnhancedChat:
    """知识图谱增强的对话系统"""
    
    def __init__(self, checkpoint_path: str, kg_path: str, device: str = "cuda"):
        print("=" * 60)
        print("  HomesteadLM-157M 知识图谱增强对话")
        print("=" * 60)
        
        # 加载知识图谱
        self.kg = KnowledgeGraph(kg_path)
        
        # 加载模型
        print(f"\n[*] 加载检查点: {checkpoint_path}")
        print(f"[*] 正在加载权重到 {device}...")
        
        ck = torch.load(checkpoint_path, map_location=device, weights_only=True)
        print("[*] 权重加载完成")
        
        self.config = ck["config"]
        self.state_dict = ck["model_state_dict"]
        self.step = ck.get("step", "?")
        
        print(f"[*] 训练步数: {self.step}")
        print(f"[*] 模型配置: hidden={self.config['hidden_size']}, "
              f"layers={self.config['num_layers']}, "
              f"experts={self.config['num_experts']}")
        
        # 创建模型
        self.model = MiniMoETiny(self.config)
        self.model.load_state_dict(self.state_dict)
        self.model.to(device)
        self.model.eval()
        
        # Tokenizer
        self.tokenizer = Tokenizer()
        print(f"[*] 词表: {self.tokenizer.vocab_size} (BBPE)")
        
        # CUDA优化
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
        
        # 预热
        print("[*] 预热推理...")
        dummy = torch.randint(0, self.config["vocab_size"], (1, 10), device=device)
        with torch.no_grad():
            _ = self.model(dummy)
        print("[*] 预热完成，可以开始对话\n")
        
        self.device = device
        self.conversation = []
    
    @torch.no_grad()
    def model_generate(self, user_text: str, max_tokens: int = 80) -> str:
        """使用模型生成回复"""
        # 构建简洁输入
        input_text = f"问: {user_text}\n答:"
        
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        
        # 生成
        generated = torch.tensor([input_ids], device=self.device)
        
        prev_tokens = set()
        stop_tokens = {self.tokenizer.eos_id}  # 只在eos停止
        
        for i in range(max_tokens):
            if generated.shape[1] > self.config.get("max_seq_len", 512):
                generated = generated[:, -self.config.get("max_seq_len", 512):]
            
            outputs = self.model(generated)
            logits = outputs["logits"][:, -1, :].float()
            
            # 重复惩罚 - 禁止生成特定token
            banned = {self.tokenizer.bos_id, 3}  # 3可能是换行等
            for tid in banned:
                logits[0, tid] = float("-inf")
            
            for tid in prev_tokens:
                logits[0, tid] /= 1.2
            
            # 采样
            probs = F.softmax(logits / 0.7, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            if next_token.item() in stop_tokens:
                break
            
            prev_tokens.add(next_token.item())
            if len(prev_tokens) > 50:
                prev_tokens = set(list(prev_tokens)[-30:])
            
            generated = torch.cat([generated, next_token], dim=1)
        
        # 解码并清理
        out_ids = generated[0, len(input_ids):].tolist()
        response = self.tokenizer.decode(out_ids, skip_special_tokens=True)
        
        # 清理格式
        response = response.replace(" ", "")
        response = response.replace("问:", "").replace("答:", "")
        response = response.replace("用户:", "").replace("助手:", "")
        response = re.sub(r'\n+', ' ', response).strip()
        
        # 截断到第一个句号或问号后
        for punct in ['。', '！', '？']:
            if punct in response and response.index(punct) < len(response) - 2:
                response = response[:response.index(punct) + 1]
        
        return response if response else "抱歉，我不太明白你的意思。"
    
    def chat(self):
        """对话循环"""
        print("=" * 60)
        print("  命令: /quit 退出 | /clear 清空 | /kg off/on 开关知识图谱")
        print("=" * 60 + "\n")
        
        kg_enabled = True
        
        while True:
            try:
                user_text = input("你: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n[*] 退出")
                break
            
            if not user_text:
                continue
            
            if user_text.lower() in ("/quit", "/exit", "/q"):
                break
            
            if user_text == "/clear":
                self.conversation = []
                print("[*] 对话历史已清空\n")
                continue
            
            if user_text.lower() == "/kg off":
                kg_enabled = False
                print("[*] 知识图谱已关闭\n")
                continue
            
            if user_text.lower() == "/kg on":
                kg_enabled = True
                print("[*] 知识图谱已开启\n")
                continue
            
            # 添加用户输入
            self.conversation.append({"role": "user", "content": user_text})
            
            print("AI: ", end="", flush=True)
            t0 = time.time()
            
            # 1. 知识图谱匹配
            if kg_enabled:
                intent, kg_response, confidence = self.kg.match(user_text)
                if confidence >= 0.5:  # 降低阈值
                    print(f"[KG] ", end="", flush=True)
                    response = kg_response
                else:
                    # 2. 模型生成
                    print("[Model] ", end="", flush=True)
                    response = self.model_generate(user_text)
            else:
                response = self.model_generate(user_text)
            
            elapsed = time.time() - t0
            print(f"\n{response}")
            print(f"[{elapsed:.1f}s]\n")
            
            # 添加助手回复
            self.conversation.append({"role": "assistant", "content": response})
            
            # 保持对话长度
            if len(self.conversation) > 10:
                self.conversation = self.conversation[-8:]


def main():
    parser = argparse.ArgumentParser(description="HomesteadLM-157M 知识图谱对话")
    parser.add_argument("--checkpoint", type=str,
                       default="checkpoints/minimoe_final.pt")
    parser.add_argument("--kg", type=str,
                       default="知识图谱/daily_chat_knowledge_graph.json")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[!] CUDA不可用，切换到CPU")
        device = "cpu"
    
    # 路径处理
    base_dir = Path(__file__).parent
    ck_path = args.checkpoint
    if not os.path.isabs(ck_path):
        ck_path = str(base_dir / ck_path)
    
    kg_path = args.kg
    if not os.path.isabs(kg_path):
        kg_path = str(base_dir / kg_path)
    
    if not os.path.exists(ck_path):
        print(f"[!] 检查点不存在: {ck_path}")
        sys.exit(1)
    
    if not os.path.exists(kg_path):
        print(f"[!] 知识图谱不存在: {kg_path}")
        sys.exit(1)
    
    # 启动
    chat = KGEnhancedChat(ck_path, kg_path, device)
    chat.chat()


if __name__ == "__main__":
    main()
