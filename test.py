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

import os
os.environ['HF_HOME'] = 'D:/work/gpt/cache/'

base_model = AutoModelForCausalLM.from_pretrained(
    'D:/work/gpt/llama-7b-hf',
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,   # optional if you have enough VRAM
    cache_dir='D:/work/gpt/cache/'
)
tokenizer = AutoTokenizer.from_pretrained('D:/work/gpt/llama-7b-hf')

model = PeftModel.from_pretrained(base_model, 'D:/work/gpt/fingpt-forecaster_dow30_llama2-7b_lora')

# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# model = model.eval()

# Make prompts
prompt = [
'''Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}
Input: FINANCING OF ASPOCOMP 'S GROWTH Aspocomp is aggressively pursuing its growth strategy by increasingly focusing on technologically more demanding HDI printed circuit boards PCBs .
Answer: ''',
'''Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}
Input: According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing .
Answer: '''
]

# Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

tokens = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
res = model.generate(**tokens, max_length=512)
res_sentences = [tokenizer.decode(i) for i in res]
out_text = [o.split("Answer: ")[1] for o in res_sentences]


# Show results
for sentiment in out_text:
    print(sentiment)