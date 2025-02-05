# DeepSeek-R1-Distill-Qwen-1.5B

# from transformers import AutoTokenizer, AutoModelForCausalLM

# model = AutoModelForCausalLM.from_pretrained("D:/work/gpt/model/DeepSeek-R1-Distill-Qwen-1.5B", trust_remote_code=True, device_map = "cuda:0", load_in_8bit = True,)
# model = model.eval()

from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "D:/work/gpt/model/DeepSeek-R1-Distill-Qwen-1.5B",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("D:/work/gpt/model/DeepSeek-R1-Distill-Qwen-1.5B")

# input question, and output answer
prompt = "Give me a short introduction to iPhone."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=2048
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("result response : ")
print(response)