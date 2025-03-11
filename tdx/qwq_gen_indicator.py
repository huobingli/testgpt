# -*- coding: utf-8 -*-

prompt_tdx = """
请基于以下指导原则根据用户实际输入为通达信编写技术分析指标公式。请确保你的回答仅适用于通达信平台，并且提供的代码可以直接在通达信平台上运行。避免使用任何同花顺特有的语法或函数。
#### 指导原则：
1. **注释方式**：
   - 使用 `{}` 进行单行或多行注释。
     ```plaintext
     {这是单行注释}
     {
     这是多行
     注释
     }
     ```
2. **变量声明与赋值**：
   - 变量声明不超过16个字符，不需要显式声明类型，但通常以 `:` 结尾来增强可读性。例如：`DIFF: EMA(CLOSE, 12) - EMA(CLOSE, 26);`
3. **变量输出**：
   - 变量声明不超过16个字符，不需要显式声明类型，但通常以 `=` 结尾来增强可读性。例如：`DIFF= EMA(CLOSE, 12) - EMA(CLOSE, 26);`
4. **逻辑运算符**：
   - 使用 `AND`, `OR`, `NOT`。虽然也支持符号形式（如 `&&`, `||`, `!`），但在实践中更多见全拼形式。
5. **函数实现**：
   - 遵循通达信的函数命名和使用规则，确保不引入同花顺特有的函数或语法，生成内容禁止复述示例，最终结果只包含用户实际输入生成的通达信代码。 
6. **注意**：
   - 示例仅用作参考,与输出无关,输出只和用户实际输入相关。

#### 示例：
用户输入：“当股价出现MACD金叉并且收盘价突破5日均线时买入”。
请按照以下格式提供答案：

```plaintext
{ 计算MACD的DIFF值 }
DIFF: EMA(CLOSE, 12) - EMA(CLOSE, 26);
{ 计算DEA值 }
DEA: EMA(DIFF, 9);
{ MACD柱状图 }
MACD: 2*(DIFF-DEA), COLORSTICK;
{ 5日均线 }
MA5: MA(CLOSE, 5);
{ 定义买入条件：MACD金叉且收盘价上穿5日均线 }
CROSS(DIFF, DEA) AND CLOSE > REF(MA5, 1);
```

####用户实际输入：
{que}"""

import requests
from apikey import chat_url

def qwq(messages):
    # chat_url = chat_url
    chat_url = 'http://localhost:11434/api/generate'
    header = 'Content-Type: application/json'
    payload = {
        # "model": "DeepSeek",
        "model": "qwen2.5-coder:32b",
        "messages": [
            {
                "role": "system",
                "content": "you are a helpful assistant."
            },
            {
                "role": "user",
                "content": messages
            }
        ],
        "max_tokens": 32000,
        "temperature": 0.5,
        "stream": False
        }
    res = requests.post(chat_url, json=payload)
    print(res.status_code)
    print(res.text)

    print(res.json())
   #  return res.json()["choices"][0]["message"]["content"]


# qwq("帮我用通达信指标体系写一个同时突破5日10日均值买入的策略代码.")


import re


def extract_code_blocks(text):
    pattern = r'```plaintext(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


que = "帮我用通达信指标体系写一个同时突破5日10日均价时买入的策略代码."
input_str = prompt_tdx.replace("{que}", que)
answer = qwq(input_str)
print(answer)