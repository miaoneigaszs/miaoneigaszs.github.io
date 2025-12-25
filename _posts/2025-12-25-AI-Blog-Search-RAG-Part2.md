---
title: "AI Blog Search - 深入代码实现与优化（下）"

categories:
  - RAG
tags:
  - LangGraph
  - RAG
---

## 前言

在[上篇](/ai/rag/AI-Blog-Search-RAG-Part1/)中，我们介绍了这个自适应 RAG 系统的核心特性和架构设计。本篇将深入代码实现，解析关键函数和优化细节。

---

## 🔐 URL 去重机制

防止重复索引是一个重要的工程问题。系统采用 **双重检查机制**：

### 1. 生成 URL 哈希

```python
def get_url_hash(url: str) -> str:
    """生成 URL 的 MD5 哈希值"""
    return hashlib.md5(url.encode()).hexdigest()
```

### 2. 数据库层面检查

```python
def check_url_exists_in_db(client: QdrantClient, url: str) -> bool:
    """检查 URL 是否已存在于数据库"""
    url_hash = get_url_hash(url)
    result = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[FieldCondition(
                key="metadata.url_hash",
                match=MatchValue(value=url_hash)
            )]
        ),
        limit=1
    )
    return len(result[0]) > 0
```

{: .notice--warning}
**为什么需要双重检查？** 内存缓存（`st.session_state`）在应用重启后会丢失，而数据库层面的检查确保了即使重启也不会重复索引。

---

## 🧠 LangGraph 状态图

系统的核心是一个状态图，定义了 Agent 的行为：

### 状态定义

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    loop_step: int      # 当前循环步数
    run_mode: str       # 运行模式：fast/deep
```

### 构建图

```python
def get_graph(retriever_tool, api_key):
    tools = [retriever_tool]
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("agent", partial(agent, tools=tools, api_key=api_key))
    workflow.add_node("retrieve", ToolNode(tools))
    workflow.add_node("rewrite", partial(rewrite, api_key=api_key))
    workflow.add_node("generate", partial(generate, api_key=api_key))

    # 定义边
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition, 
                                   {"tools": "retrieve", END: END})
    workflow.add_conditional_edges("retrieve", 
                                   partial(grade_documents, api_key=api_key))
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")

    return workflow.compile()
```

---

## 🔎 智能评分机制

评分节点决定是生成答案还是重写查询：

```python
def grade_documents(state, api_key) -> Literal["generate", "rewrite"]:
    mode = state.get("run_mode", "deep")
    
    # 快速模式直接跳过评分
    if mode == "fast":
        return "generate"

    # 达到最大重试次数，强制生成
    current_step = state.get("loop_step", 0)
    if current_step >= 3:
        return "generate"

    # 使用结构化输出进行评分
    class Grade(BaseModel):
        binary_score: str = Field(description="'yes' or 'no'")

    prompt = ChatPromptTemplate.from_template(
        """你是评分员。评估检索片段是否能回答问题。
        
        规则：
        1. 如果片段包含问题中提到的概念/术语的定义，评为 'yes'
        2. 即使用词不完全匹配，但语义相关，也评为 'yes'
        3. 只有完全无关时才评为 'no'
        
        问题: {question}
        片段: {context}
        
        评分 (yes/no):"""
    )
    # ...
```

{: .notice--info}
**关键优化**：评分提示词经过精心设计，采用宽容策略，只有完全无关时才会触发重写。

---

## ✍️ 智能查询重写

当评分为 "no" 时，系统会重写查询：

```python
def rewrite(state, api_key):
    question = state["messages"][0].content
    current_step = state.get("loop_step", 0)
    
    prompt = ChatPromptTemplate.from_template(
        """你的任务是重写问题以提高检索效果。
        
        原问题: {question}
        
        重写规则：
        - 保留核心概念词
        - 用多个同义词表达（如"概念"可以说成"定义、含义、解释"）
        - 简化为陈述句形式
        
        重写后的问题:"""
    )
    
    chain = prompt | get_llm(api_key) | StrOutputParser()
    rewritten = chain.invoke({"question": question})
    
    return {
        "messages": [HumanMessage(content=rewritten)], 
        "loop_step": current_step + 1  # 增加步数计数
    }
```

---

## 📊 MMR 检索优化

使用最大边际相关性（MMR）检索，平衡相关性和多样性：

```python
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 10,           # 返回 10 个结果
        "fetch_k": 20,     # 初始检索 20 个候选
        "lambda_mult": 0.7 # 相关性权重 0.7，多样性权重 0.3
    }
)
```

---

## 🔮 未来改进方向

### 1. 多 Agent 协作

```
Router Agent → 分类问题类型
    ├─ RAG Agent      → 知识库问答
    ├─ Search Agent   → 实时搜索
    └─ Summary Agent  → 文档摘要
```

### 2. Tool Calling

- 集成外部工具（搜索引擎、计算器、代码执行）
- 让 Agent 自主决定何时调用工具

### 3. 多轮对话

- 对话历史管理
- 上下文压缩（长对话时）
- 指代消解

### 4. RAG 评估

- 集成 RAGAS 评估框架
- 评估指标：Faithfulness、Answer Relevancy、Context Precision

### 5. 流式输出

- Streaming 实时显示生成内容
- 提升用户体验

---

## 💡 总结

这个项目展示了如何使用 LangGraph 构建一个生产级的 RAG 系统。核心要点：

1. **Agent 循环** - 而非简单的链式调用
2. **自适应重写** - 提升检索召回率
3. **双重去重** - 内存 + 数据库层面保障
4. **中文优化** - BGE 模型 + 语义切分

<details markdown="1">
<summary>点击查看完整源码</summary>

📥 **[下载 main.py 源码](/assets/Codes/main.py)**

源码包含约 500 行 Python 代码，涵盖：
- Qdrant 向量数据库初始化
- LangGraph 状态图构建
- 检索、评分、重写、生成节点实现
- Streamlit Web UI

</details>
