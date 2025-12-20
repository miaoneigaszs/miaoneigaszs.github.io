---
title: Agent Use Search Tool

categories:
  - Simple Agent
tags:
  - Simple Agent
---

## 搭建一个最简单的使用搜索工具的Agent，使用agno框架，搜索工具使用``baidusearch``、``duckduckgo``、``tavily``。

构建Agent：
```python
llm=OpenAILike(
    id='deepseek-chat',
    api_key='your_api_key',
    base_url='https://api.deepseek.chat/v1'
)
agent = Agent(
    name="search_tool_agent",
    model=llm,
    tools=['baidusearch', 'duckduckgo', 'tavily'],
    description="你是一个AI聊天机器人,必要时需要调用搜索工具辅助回答",
    instructions=[
        "你是一个AI聊天机器人,必要时需要调用搜索工具辅助回答",
        tool_instruction
    ]
)
```

回答和流式输出：
```python
response : RunOutput = agent.run(input_text)

for chunk in response:
    if chunk.content:
        print(chunk.content)
```

演示效果：
{: .notice--info} 

![演示图片](/assets/images/agent-search-tool.png)

<details markdown="1">
<summary>点击此处查看源码</summary>

```python

import streamlit as st
from dotenv import load_dotenv
import os
from agno.models.openai import OpenAIChat, OpenAILike
from agno.agent import Agent
from agno.run.agent import RunOutput
from agno.tools.baidusearch import BaiduSearchTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.tavily import TavilyTools

load_dotenv()

st.set_page_config(page_title = "🤔Try Agno")

st.title("🔎结合搜索引擎工具的AI")


openai_key = os.getenv("LLM_API_KEY")
base_url = os.getenv("LLM_BASE_URL")
model_id = os.getenv("LLM_MODEL_ID")


llm = OpenAILike(
    api_key = openai_key,
    base_url = base_url,
    id = model_id
)

st.sidebar.header("🛠️选择工具")
tool_name = st.sidebar.radio(
    "辅助工具",
    ["百度搜索 (中文)", "DuckDuckGo (国际)", "Tavily(SaaS服务)"]
)


selected_tools = []
tool_instruction = ""

if tool_name == "百度搜索 (中文)":
    selected_tools = [BaiduSearchTools()]
    tool_instruction = "优先使用百度搜索中文信息，回答必须注明来源。"
elif tool_name == "DuckDuckGo (国际)":
    selected_tools = [DuckDuckGoTools()]
    tool_instruction = "使用 DuckDuckGo 搜索国际互联网信息。"
else:
    selected_tools  = [TavilyTools()]
    tool_instruction = "使用Tavily搜索信息，回答必须注明来源。"
    

st.header("🤔试试聊天吧")

input_text = st.text_input("输入聊天内容：")

agent = Agent(
    name = "chatbot_use_tools",
    model = llm,
    tools = selected_tools, 
    description = "你是一个AI聊天机器人,必要时需要调用搜索工具辅助回答",
    instructions = [
        "你是一个AI聊天机器人,必要时需要调用搜索工具辅助回答",
        tool_instruction
    ],
)

if input_text:
    # 👇 1. 创建一个空的容器，用于动态显示内容
    response_placeholder = st.empty()
    full_response = ""

    # 👇 2. 开启 stream=True，进入循环
    try:
        response_generator = agent.run(input_text, stream=True)
        
        for chunk in response_generator:
            # 有些 chunk 可能是工具调用的过程信息，只有 content 才是回答文本
            if chunk.content:
                full_response += chunk.content
                # 实时更新网页上的内容，"▌" 是光标效果
                response_placeholder.markdown(full_response + "▌")
        
        # 循环结束，把最后的光标去掉
        response_placeholder.markdown(full_response)
        
    except Exception as e:
        st.error(f"发生错误: {e}")
```
</details>