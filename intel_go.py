from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForQuestionAnswering

model = ORTModelForQuestionAnswering.from_pretrained("D:/work/gpt/llama-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("D:/work/gpt/llama-7b-hf")

onnx_qa = pipeline("question-answering", model=model, tokenizer=tokenizer)

question = "What's my name?"
context = "My name is Philipp and I live in Nuremberg."
pred = onnx_qa(question, context)