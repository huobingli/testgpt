import torch
# from transformers import LLaMAForCausalLM, LLaMATokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型和分词器
model_name = 'D:/work/gpt/llama-7b-hf'  # 确保路径正确
tokenizer = LLaMATokenizer.from_pretrained(model_name)
model = LLaMAForCausalLM.from_pretrained(model_name)

# 将模型移动到 CPU
device = torch.device('cpu')
model.to(device)

# 对输入进行分词
input_text = "Hello, how are you?"
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

# 生成输出
with torch.no_grad():
    output_ids = model.generate(input_ids, max_length=50)

# 解码输出
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)