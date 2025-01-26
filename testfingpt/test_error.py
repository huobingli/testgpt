
# # Load Models
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
# model = AutoModelForCausalLM.from_pretrained("D:/work/gpt/testgpt/testfingpt/fingpt-mt_llama2-7b_lora")

# model, tokenizer = load_model(base_model, peft_model, FROM_REMOTE)


# demo_tasks = [
#     'Financial Sentiment Analysis',
#     'Financial Relation Extraction',
#     'Financial Headline Classification',
#     'Financial Named Entity Recognition',
# ]
# demo_inputs = [
#     "Glaxo's ViiV Healthcare Signs China Manufacturing Deal With Desano",
#     "Apple Inc Chief Executive Steve Jobs sought to soothe investor concerns about his health on Monday, saying his weight loss was caused by a hormone imbalance that is relatively simple to treat.",
#     'gold trades in red in early trade; eyes near-term range at rs 28,300-28,600',
#     'This LOAN AND SECURITY AGREEMENT dated January 27 , 1999 , between SILICON VALLEY BANK (" Bank "), a California - chartered bank with its principal place of business at 3003 Tasman Drive , Santa Clara , California 95054 with a loan production office located at 40 William St ., Ste .',
# ]
# demo_instructions = [
#     'What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}.',
#     'Given phrases that describe the relationship between two words/phrases as options, extract the word/phrase pair and the corresponding lexical relationship between them from the input text. The output format should be "relation1: word1, word2; relation2: word3, word4". Options: product/material produced, manufacturer, distributed by, industry, position held, original broadcaster, owned by, founded by, distribution format, headquarters location, stock exchange, currency, parent organization, chief executive officer, director/manager, owner of, operator, member of, employer, chairperson, platform, subsidiary, legal form, publisher, developer, brand, business division, location of formation, creator.',
#     'Does the news headline talk about price going up? Please choose an answer from {Yes/No}.',
#     'Please extract entities and their types from the input sentence, entity types should be chosen from {person/organization/location}.',
# ]

# test_demo(model, tokenizer)


from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizerFast, LlamaForCausalLM
from peft import PeftModel

# hf_token = "xxx" #Put your own HF token here, do not publish it
# from huggingface_hub import login
# # Login directly with your Token (remember not to share this Token publicly)
# login(token=hf_token)

# base test model
# base_model = AutoModelForCausalLM.from_pretrained(
#     'D:/work/gpt/model/Llama-2-7b-chat-hf',
#     trust_remote_code=True,
#     device_map = "cuda:0",
#     torch_dtype=torch.float16,   # optional if you have enough VRAM
# )
# tokenizer = AutoTokenizer.from_pretrained('D:/work/gpt/model/Llama-2-7b-chat-hf')

# model = PeftModel.from_pretrained(base_model, 'D:/work/gpt/fingpt-forecaster_dow30_llama2-7b_lora')
# model = model.eval()

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")
# torch.set_default_device(device)

# github ipynb model test
base_model = "D:/work/gpt/model/Llama-2-7b-chat-hf"
peft_model = "D:/work/gpt/fingpt-forecaster_dow30_llama2-7b_lora"

tokenizer = LlamaTokenizerFast.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
# model = LlamaForCausalLM.from_pretrained(base_model, trust_remote_code=True, device_map = "cuda:0", load_in_8bit = True,)
model = LlamaForCausalLM.from_pretrained(
    base_model,
    trust_remote_code=True,
    device_map="cpu",                       # 强制使用CPU
    torch_dtype=torch.float16,              # 仍然使用fp16以节省内存
)

model = PeftModel.from_pretrained(model, peft_model, torch_dtype=torch.float16, device_map="cpu")
# model = model.cuda()
model = model.eval()

# Make prompts
prompt = [
'''Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}
Input: FINANCING OF ASPOCOMP 'S GROWTH Aspocomp is aggressively pursuing its growth strategy by increasingly focusing on technologically more demanding HDI printed circuit boards PCBs .
Answer: ''',
'''Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}
Input: According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing .
Answer: ''',
'''Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}
Input: A tinyurl link takes users to a scamming site promising that users can earn thousands of dollars by becoming a Google ( NASDAQ : GOOG ) Cash advertiser .
Answer: ''',
]

# Generate results
tokens = tokenizer(prompt, return_tensors='pt', padding=True, max_length=512)
res = model.generate(**tokens, max_length=512)
res_sentences = [tokenizer.decode(i) for i in res]
out_text = [o.split("Answer: ")[1] for o in res_sentences]

# show results
for sentiment in out_text:
    print(sentiment)

# Output:
# positive
# neutral
# negative
