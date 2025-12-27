---
title: "实战：搭建基于 MCP 的浏览器 Agent"
date: 2025-12-20
categories:
  - Technology
tags:
  - MCP
  - Browser
date: 2025-12-20 17:00:00 +0800
toc: true
toc_sticky: true
toc_label: "文章目录"
toc_icon: "book"
excerpt: "本文介绍如何使用 Python 和 Playwright 搭建一个支持自然语言控制的浏览器 Agent，并解决了 DeepSeek API 的兼容性问题。"
---

## 前言

接着上一篇关于 MCP 基础知识的介绍，今天我们来实战搭建一个浏览器 MCP Agent。

我们将构建一个可以用自然语言控制浏览器的智能体。该项目使用了 **Playwright** 作为 MCP 工具，并使用 `mcp-agent` 库来构建客户端。

## 演示视频

<video width="100%" controls>
  <source src="/assets/videos/MCP-Browser.mp4" type="video/mp4">
  您的浏览器不支持 Video 标签。
</video>

*演示：Agent 自动打开网页并执行操作*

---

## 踩坑：DeepSeek API 的兼容性问题

在构建 Client 的初期，我尝试使用 DeepSeek 的 API。虽然 DeepSeek 号称兼容 OpenAI 格式，但在实际对接 Agent 框架时，我遇到了一个棘手的兼容性问题。

### 问题描述
当 Agent 调用工具（如抓取网页）并将结果存入历史记录（History）时，`mcp-agent` 库会生成一种复杂的 List 结构（包含文本类型的 Metadata 和截图数据等）。

### 失败原因
DeepSeek 的 API 对 `messages` 格式的检查非常严格。它期望历史消息是纯字符串（String），无法解析这种复杂的序列结构（Sequence），导致请求直接报错。

### 解决方案
最终我改用了中转站的 ChatGPT API。由于原生 OpenAI 接口对这种多模态/结构化数据的包容性更好，问题迎刃而解。

**经验总结：** 在开发 Agent 时，如果遇到工具调用（Function Calling）报错，优先检查模型 API 对 History 格式的兼容性。
{: .notice--warning}

---

## 搭建 MCP Client 实战

搭建一个基于 `mcp-agent` 的客户端主要涉及配置文件的编写和核心代码的初始化。

### 1. 配置文件准备

首先，我们需要一个配置文件来告诉 Client 去哪里找 Server。

**注意：** 文件名必须严格命名为 `mcp_agent.config.yaml`，不能随意修改（例如写成 `mcp_agent_config.yaml`），否则程序无法自动加载默认配置。
{: .notice--danger}

在项目根目录下创建 `mcp_agent.config.yaml`：

```yaml
execution_engine: asyncio
logger:
  level: info

mcp:
  servers:
    # 定义 playwright 服务，使用 npx 启动
    playwright:
      command: "npx"
      # -y 防止第一次运行询问，同时伪装 User-Agent 防止被反爬
      args: 
        - "-y"
        - "@playwright/mcp@latest"
        - "--"
        - "--disable-blink-features=AutomationControlled"

openai:
  # 根据你的实际 API 提供商填写模型名称
  default_model: "gpt-4o-mini" 
```

### 2. 核心变量解析

在 Python 代码中（结合 Streamlit），我们需要初始化几个关键变量，它们构成了 MCP Client 的生命周期。

* **`mcp_app`**: `MCPApp(name='browser_agent')`
  * 这是整个应用的“容器”或“身份证”。它负责读取本地配置，定义应用的名称，但此时连接尚未建立。

* **`mcp_context`**: 连接管理器
  * 这是通过 `app.run()` 获得的上下文对象。它负责管理到底层 Server 的物理连接（Socket 或 Stdio）。它就像是“拨通电话”的动作。

* **`mcp_agent_app`**: 应用的运行时实例
  * 这是真正“通电”后的应用状态。它维护着当前可用的工具列表和服务器资源池。

* **`agent`**: 智能体 (LLM 模块)
  * 这是具体的业务逻辑执行者。它绑定了 LLM（大脑）和 Server（手脚）。

---

## 完整代码实现

以下是基于 Streamlit 实现的完整 Python 代码。它实现了界面的渲染、Agent 的异步初始化以及指令的执行。

<!-- 使用 details 标签来折叠长代码 -->
<details markdown="1">
<summary><strong>📄 点击展开/折叠：完整 Python 实现代码 (main.py)</strong></summary>

```python
import asyncio
import os
import streamlit as st
from textwrap import dedent
from dotenv import load_dotenv

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams

load_dotenv()

# Page config
st.set_page_config(page_title="Browser MCP Agent", page_icon="🌐", layout="wide")

# Title and description
st.markdown("<h1 class='main-header'>🌐 Browser MCP Agent</h1>", unsafe_allow_html=True)
st.markdown("Interact with a powerful web browsing agent that can navigate and interact with websites")

# Setup sidebar with example commands
with st.sidebar:
    st.markdown("### Example Commands")
    
    st.markdown("**Navigation**")
    st.markdown("- Go to [github.com/Shubhamsaboo/awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps)")
    
    st.markdown("**Interactions**")
    st.markdown("- click on mcp_ai_agents")
    st.markdown("- Scroll down to view more content")
    
    st.markdown("**Multi-step Tasks**")
    st.markdown("- Navigate to [github.com/Shubhamsaboo/awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps), scroll down, and report details")
    st.markdown("- Scroll down and summarize the github readme")
    
    st.markdown("---")
    st.caption("Note: The agent uses Playwright to control a real browser.")

# Query input
query = st.text_area("Your Command", 
                   placeholder="Ask the agent to navigate to websites and interact with them")

# Initialize app and agent
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.mcp_app = MCPApp(name="streamlit_mcp_agent")
    st.session_state.mcp_context = None
    st.session_state.mcp_agent_app = None
    st.session_state.browser_agent = None
    st.session_state.llm = None
    st.session_state.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(st.session_state.loop)
    st.session_state.is_processing = False

# Setup function that runs only once
async def setup_agent():
    if not st.session_state.initialized:
        try:
            # Create context manager and store it in session state
            st.session_state.mcp_context = st.session_state.mcp_app.run()
            st.session_state.mcp_agent_app = await st.session_state.mcp_context.__aenter__()
            
            # Create and initialize agent
            st.session_state.browser_agent = Agent(
                name="browser",
                instruction="""You are a helpful web browsing assistant that can interact with websites using playwright.
                    - Navigate to websites and perform browser actions (click, scroll, type)
                    - Extract information from web pages 
                    - Provide concise summaries of web content using markdown
                    - Follow multi-step browsing sequences to complete tasks
                    - Do NOT take screenshots.
                    - Do NOT use any image tools.
                    
                Respond back with a status update on completing the commands.""",
                server_names=["playwright"],
            )
            
            # Initialize agent and attach LLM
            await st.session_state.browser_agent.initialize()
            st.session_state.llm = await st.session_state.browser_agent.attach_llm(OpenAIAugmentedLLM)
            
            # List tools once
            logger = st.session_state.mcp_agent_app.logger
            tools = await st.session_state.browser_agent.list_tools()
            logger.info("Tools available:", data=tools)
            
            # Mark as initialized
            st.session_state.initialized = True
        except Exception as e:
            return f"Error during initialization: {str(e)}"
    return None

# Main function to run agent
async def run_mcp_agent(message):
    print(f"DEBUG: 开始处理消息 - {message}") # 调试信息
    if not os.getenv("OPENAI_API_KEY"):
        return "Error: OpenAI API key not provided"
    
    try:
        error = await setup_agent()
        if error:
            print(f"DEBUG: Setup 失败 - {error}") 
            return error
        
        print("DEBUG: Agent Setup 成功，准备发送给 LLM...") 
        
        # 你的 LLM 生成代码
        result = await st.session_state.llm.generate_str(
            message=message, 
            request_params=RequestParams(use_history=True, maxTokens=10000)
            )

        
        print(f"DEBUG: LLM 返回结果类型: {type(result)}") 
        print(f"DEBUG: LLM 返回内容: {result}")          
        
        return result
        
    except Exception as e:
        print(f"DEBUG: 发生严重异常: {str(e)}") 
        import traceback
        traceback.print_exc() # 👈 把详细报错打印出来
        return f"Error: {str(e)}"

# Defaults
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'last_result' not in st.session_state:
    st.session_state.last_result = None

def start_run():
    st.session_state.is_processing = True

# Button (use a callback so the click just flips state)
st.button(
    "🚀 Run Command",
    type="primary",
    use_container_width=True,
    disabled=st.session_state.is_processing,
    on_click=start_run,
)

# If we’re in a processing run, do the work now
if st.session_state.is_processing:
    with st.spinner("Processing your request..."):
        result = st.session_state.loop.run_until_complete(run_mcp_agent(query))
    # persist result across the next rerun
    st.session_state.last_result = result
    # unlock the button and refresh UI
    st.session_state.is_processing = False
    st.rerun()

# Render the most recent result (after the rerun)
if st.session_state.last_result:
    st.markdown("### Response")
    st.markdown(st.session_state.last_result)
else:
    # (your existing help text here)
    pass

# Display help text for first-time users
if 'result' not in locals():
    st.markdown(
        """<div style='padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
        <h4>How to use this app:</h4>
        <ol>
            <li>Enter your OpenAI API key in your mcp_agent.secrets.yaml file</li>
            <li>Type a command for the agent to navigate and interact with websites</li>
            <li>Click 'Run Command' to see results</li>
        </ol>
        <p><strong>Capabilities:</strong></p>
        <ul>
            <li>Navigate to websites using Playwright</li>
            <li>Click on elements, scroll, and type text</li>
            <li>Take screenshots of specific elements</li>
            <li>Extract information from web pages</li>
            <li>Perform multi-step browsing tasks</li>
        </ul>
        </div>""", 
        unsafe_allow_html=True
    )

# Footer
st.markdown("---")
st.write("Built with Streamlit, Playwright, and [MCP-Agent](https://www.github.com/lastmile-ai/mcp-agent) Framework ❤️")
```

</details>

## 为什么选择 MCP？

在这次搭建过程中，我深刻体会到了 MCP 架构带来的优势：

1. **解耦与安全**：工具（Playwright）运行在独立的 Node.js 进程中，与我的 Python 主程序分离。即便工具崩溃或受到攻击，也不会直接影响主进程的环境。
2. **灵活性**：我不需要在 Python 代码里硬编码浏览器操作的逻辑，只需要通过协议发送“意图”，由 Server 去执行具体的脏活累活。

这种不再与客户端强绑定的工具调用方式，正是未来 AI Agent 开发的主流方向。（这篇文章比较长，我借助了LLM帮我写了初稿，后续会继续完善。）
{: .notice--primary}
