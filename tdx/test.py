import requests

url = "http://localhost:11434/api/tags"
response = requests.get(url)

print(response.json())  # 返回本地所有可用的模型