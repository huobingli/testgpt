from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import LoraConfig, get_peft_model

# 选择模型
model_name = "D:/work/gpt/model/DeepSeek-R1-Distill-Qwen-1.5B"

# 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 4-bit 量化加载模型（适用于低显存）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # 自动分配 GPU/CPU
    load_in_4bit=True,  # 4-bit 量化
    torch_dtype=torch.float16
)

# 配置 QLoRA
lora_config = LoraConfig(
    r=8,  # 低秩矩阵，较小的值适用于低显存
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # 适配 Transformer 注意力层
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 应用 LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()