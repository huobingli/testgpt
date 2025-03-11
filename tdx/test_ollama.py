from ollama import chat
from ollama import ChatResponse

response: ChatResponse = chat(model='qwq', messages=[
  {
    'role': 'user',
    'content': '为什么天空是蓝色的？',
  },
])
print(response['message']['content'])

print(response.message.content)