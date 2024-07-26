import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
def Ask(text):
    input_text = text;
    inputs = tokenizer(input_text, return_tensors="pt")
    # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    generate_ids = model.generate(inputs.input_ids, max_length=100)
    print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

tokenizer = AutoTokenizer.from_pretrained("D:/work/gpt/llama-7b-hf")
### ERROR - transformers.tokenization_utils -   Using pad_token, but it is not set yet
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("D:/work/gpt/llama-7b-hf")
# convert the model to BetterTransformer
model.to_bettertransformer()