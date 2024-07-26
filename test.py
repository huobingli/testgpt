# import torch
# from peft import PeftModel, PeftConfig
# from transformers import AutoModelForCausalLM

# config = PeftConfig.from_pretrained("D:/work/gpt/fingpt-forecaster_dow30_llama2-7b_lora")
# base_model = AutoModelForCausalLM.from_pretrained("D:/work/gpt/llama-7b-hf")
# model = PeftModel.from_pretrained(base_model, "D:/work/gpt/fingpt-forecaster_dow30_llama2-7b_lora")


# device = torch.device('cpu')
# model.to(device)
# model = model.eval()


import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


base_model = AutoModelForCausalLM.from_pretrained(
    'D:/work/gpt/llama-7b-hf',
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,   # optional if you have enough VRAM
)
tokenizer = AutoTokenizer.from_pretrained('D:/work/gpt/llama-7b-hf')

model = PeftModel.from_pretrained(base_model, 'D:/work/gpt/fingpt-forecaster_dow30_llama2-7b_lora')

device = torch.device('cpu')
model.to(device)
model = model.eval()