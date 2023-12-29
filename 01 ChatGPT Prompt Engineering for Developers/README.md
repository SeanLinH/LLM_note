# 1/2 ChatGPT Prompt Engineering for Developers(第六組)

[TOC]

---

## 課程簡介

**課程連結**: https://learn.deeplearning.ai/chatgpt-prompt-eng/lesson/1/introduction

如何有效地利用ChatGPT進行對話式AI的訓練和應用。涵蓋了如何引導和改善ChatGPT的回應以達到特定的目標或性能。以下主題：

- [**Introduction**](#Introduction): 介紹ChatGPT和對話式AI的基礎。
- [**Guidelines**](#Guidelines): 提供使用和訓練對話式AI的最佳實踐步驟。
- **Iterative**: 如何進行迭代開發，包括測試和改進對話模型。
- **Summarizing**: 教導如何訓練模型以提煉和總結信息。
- **Inferring**: 如何增強模型的推理能力。
- **Transforming**: 技巧用於轉換對話和生成特定格式的回應。
- **Expanding**: 如何擴展模型的知識基礎和回應能力。
- **Chatbot**: 深入聊天機器人的建構和特定用途。
- **Conclusion**: 總結課程的重點和學習成果。
- **Course Feedback**: 收集對課程的反饋以進行改進。
- **Community**: 建立一個學研社群，促進知識和經驗的分享。


## Introduction
:::info
這裡的大型語言模型有兩種，Base LLM(基石模型)、Instruction Turned LLM，李宏毅課程講到基石模型就是在網路不斷地學，但他不一定能夠真的回答到人類的問題，這堂課主要專注在Instruction Tuned LLM。
:::

![image](https://hackmd.io/_uploads/rJiKp-YPp.png)


## Guidelines

:::info
Principle 1: 明確你的指令 Write clear and specific instructions
Principle 2: 多給模型推導機會 Give the model time to “think”
:::


## 關鍵概念與定義
### 基本操作Open API

Step 1 
在Openai 官網上註冊並進入API
![image](https://hackmd.io/_uploads/BJNKwTqPT.png)

Step 2
點擊左欄的 'API keys' 可以新增一組KEY，然後會看到一串 sk-XXXX的字樣，這個複製後記得保留，關掉視窗就不會再顯示了。


![Screenshot 2023-12-28 at 18.05.02](https://hackmd.io/_uploads/HycCd65PT.png)
![upload_4c45d5ba671088a40811aac52a70e21e-2](https://hackmd.io/_uploads/r1BOKTqPa.png)


將獲得的API_KEY 放到函式中
```python
import openai
import os

openai.api_key  = "sk-xxxx"
# openai.api_key  = os.getenv('OPENAI_API_KEY')
```


這裡有兩種方式可以使用 Open API Key

:::warning
課程的內容可能在本機端或 Colab 上會有錯誤，原因是版本問題，以下是修正後的樣子
:::

方法一 (課程的方法)

```python
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content
```

方法二 (參考openai documents)

```python 
from openai import OpenAI
client = OpenAI(
    api_key=os.getenv('API_KEY'))

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content
```


**Prompting Principles**
:::info
Principle 1: 明確你的指令 (Principle 1: Write clear and specific instructions)
Principle 2: 多給模型推導機會 (Give the model time to “think”)
:::


### 明確你的指令 (Principle 1: Write clear and specific instructions)
Tactic 1: Use delimiters to clearly indicate distinct parts of the input
Delimiters can be anything like: \`\`\`, """, < >, <tag> </tag>, :
善用分隔符號，可以更清楚表達給予模型的指示

```python
text = f"""
You should express what you want a model to do by \ 
providing instructions that are as clear and \ 
specific as you can possibly make them. \ 
This will guide the model towards the desired output, \ 
and reduce the chances of receiving irrelevant \ 
or incorrect responses. Don't confuse writing a \ 
clear prompt with writing a short prompt. \ 
In many cases, longer prompts provide more clarity \ 
and context for the model, which can lead to \ 
more detailed and relevant outputs.
"""
prompt = f"""
Summarize the text delimited by triple backticks \ 
into a single sentence.
```{text}```
"""
response = get_completion(prompt)
print(response)
```
輸出 ->
To guide a model towards the desired output and reduce irrelevant or incorrect responses, it is important to provide clear and specific instructions, which can be achieved through longer prompts that offer more clarity and context.

Tactic 2: Ask for a structured output
要求模型給予特定的結構例如 html, json

```python 
prompt = f"""
Generate a list of three made-up book titles along \ 
with their authors and genres. 
Provide them in JSON format with the following keys: 
book_id, title, author, genre.
"""
response = get_completion(prompt)
print(response)
```
輸出 ->
```json
{
  "books": [
    {
      "book_id": 1,
      "title": "The Enigma of Elysium",
      "author": "Evelyn Sinclair",
      "genre": "Mystery"
    },
    {
      "book_id": 2,
      "title": "Whispers in the Wind",
      "author": "Nathaniel Blackwood",
      "genre": "Fantasy"
    },
    {
      "book_id": 3,
      "title": "Echoes of the Past",
      "author": "Amelia Hart",
      "genre": "Romance"
    }
  ]
}
```

Tactic 3: Ask the model to check whether conditions are satisfied
要求模型給出條件步驟

```python
text_1 = f"""
Making a cup of tea is easy! First, you need to get some \ 
water boiling. While that's happening, \ 
grab a cup and put a tea bag in it. Once the water is \ 
hot enough, just pour it over the tea bag. \ 
Let it sit for a bit so the tea can steep. After a \ 
few minutes, take out the tea bag. If you \ 
like, you can add some sugar or milk to taste. \ 
And that's it! You've got yourself a delicious \ 
cup of tea to enjoy.
"""
prompt = f"""
You will be provided with text delimited by triple quotes. 
If it contains a sequence of instructions, \ 
re-write those instructions in the following format:

Step 1 - ...
Step 2 - …
…
Step N - …

If the text does not contain a sequence of instructions, \ 
then simply write \"No steps provided.\"

\"\"\"{text_1}\"\"\"
"""
response = get_completion(prompt)
print("Completion for Text 1:")
print(response)
```
輸出後 -> 
Completion for Text 1:
Step 1 - Get some water boiling.
Step 2 - Grab a cup and put a tea bag in it.
Step 3 - Once the water is hot enough, pour it over the tea bag.
Step 4 - Let the tea steep for a bit.
Step 5 - After a few minutes, take out the tea bag.
Step 6 - Add sugar or milk to taste, if desired.
Step 7 - Enjoy your delicious cup of tea.


Tactic 4: "Few-shot" prompting

```python 
prompt = f"""
Your task is to answer in a consistent style.

<child>: Teach me about patience.

<grandparent>: The river that carves the deepest \ 
valley flows from a modest spring; the \ 
grandest symphony originates from a single note; \ 
the most intricate tapestry begins with a solitary thread.

<child>: Teach me about resilience.
"""
response = get_completion(prompt)
print(response)
```
輸出 ->
\<grandparent>: Resilience is like a mighty oak tree that withstands the strongest storms, bending but never breaking. It is the ability to bounce back from adversity, to find strength in the face of challenges, and to persevere even when the odds seem insurmountable. Just as a diamond is formed under immense pressure, resilience is forged through the trials and tribulations of life.

### Principle 2: Give the model time to “think”

**Tactic 1: Specify the steps required to complete a task**
特定任務的步驟，給予模型任務的步驟更易於引導模型往正確的方向推論。
例如： 1. xxx 2. xxx 3. xxx 等，明確指示優先順序

```python 
text = f"""
In a charming village, siblings Jack and Jill set out on \ 
a quest to fetch water from a hilltop \ 
well. As they climbed, singing joyfully, misfortune \ 
struck—Jack tripped on a stone and tumbled \ 
down the hill, with Jill following suit. \ 
Though slightly battered, the pair returned home to \ 
comforting embraces. Despite the mishap, \ 
their adventurous spirits remained undimmed, and they \ 
continued exploring with delight.
"""
# example 1
prompt_1 = f"""
Perform the following actions: 
1 - Summarize the following text delimited by triple \
backticks with 1 sentence.
2 - Translate the summary into French.
3 - List each name in the French summary.
4 - Output a json object that contains the following \
keys: french_summary, num_names.

Separate your answers with line breaks.

Text:
```{text}```
"""
response = get_completion(prompt_1)
print("Completion for prompt 1:")
print(response)
```

要求特定的格式輸出這個環節也是很重要，在prompt 的部分加上 `Use the following format` 等字樣，讓模型參照開發者特定的格式。
```python 
prompt_2 = f"""
Your task is to perform the following actions: 
1 - Summarize the following text delimited by 
  <> with 1 sentence.
2 - Translate the summary into French.
3 - List each name in the French summary.
4 - Output a json object that contains the 
  following keys: french_summary, num_names.

Use the following format:
Text: <text to summarize>
Summary: <summary>
Translation: <summary translation>
Names: <list of names in summary>
Output JSON: <json with summary and num_names>

Text: <{text}>
"""
response = get_completion(prompt_2)
print("\nCompletion for prompt 2:")
print(response)
```
Completion for prompt 2:
Summary: Jack and Jill, siblings from a charming village, go on a quest to fetch water from a hilltop well, but encounter misfortune when Jack trips on a stone and tumbles down the hill, with Jill following suit, yet they remain undeterred and continue exploring with delight.

Translation: Jack et Jill, frère et sœur d'un charmant village, partent à la recherche d'eau d'un puits au sommet d'une colline, mais rencontrent un malheur lorsque Jack trébuche sur une pierre et dévale la colline, suivi de Jill, pourtant ils restent déterminés et continuent à explorer avec joie.

Names: Jack, Jill

Output JSON: 
{
  "french_summary": "Jack et Jill, frère et sœur d'un charmant village, partent à la recherche d'eau d'un puits au sommet d'une colline, mais rencontrent un malheur lorsque Jack trébuche sur une pierre et dévale la colline, suivi de Jill, pourtant ils restent déterminés et continuent à explorer avec joie.",
  "num_names": 2
}


**Tactic 2: Instruct the model to work out its own solution before rushing to a conclusion**
指示模型在急於下結論之前找出自己的解決方案




```python
prompt = f"""
Determine if the student's solution is correct or not.

Question:
I'm building a solar power installation and I need \
 help working out the financials. 
- Land costs $100 / square foot
- I can buy solar panels for $250 / square foot
- I negotiated a contract for maintenance that will cost \ 
me a flat $100k per year, and an additional $10 / square \
foot
What is the total cost for the first year of operations 
as a function of the number of square feet.

Student's Solution:
Let x be the size of the installation in square feet.
Costs:
1. Land cost: 100x
2. Solar panel cost: 250x
3. Maintenance cost: 100,000 + 100x
Total cost: 100x + 250x + 100,000 + 100x = 450x + 100,000
"""
response = get_completion(prompt)
print(response)
```

學生的答案不見得是對的，怎麼辦？
透過一些指示，指導模型一步一步完成

```
To solve the problem do the following:
- First, work out your own solution to the problem including the final total. 
- Then compare your solution to the student's solution \ 
and evaluate if the student's solution is correct or not. 
Don't decide if the student's solution is correct until 
you have done the problem yourself.
```





## 實際應用和案例研究




## 摘要與思維導圖



## 問題討論與反思




