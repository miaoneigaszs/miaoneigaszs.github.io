import os
import hashlib
import streamlit as st
from uuid import uuid4
from functools import partial
from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict

# --- 核心依赖 ---
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, Filter, FieldCondition, MatchValue
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


# --- 配置 ---
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 🔥 关键修改1: 新的集合名
COLLECTION_NAME = "rag_bge_fixed_v2" 
VECTOR_SIZE = 512  

st.set_page_config(page_title="网站检录与问答RAG", page_icon="🔧", layout="wide")

# --- 状态定义 ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    loop_step: int
    run_mode: str
    
qdrant_host = os.getenv("QDRANT_HOST")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")


# --- Session State ---
if 'qdrant_host' not in st.session_state: st.session_state.qdrant_host = qdrant_host
if 'qdrant_api_key' not in st.session_state: st.session_state.qdrant_api_key = qdrant_api_key
if 'openai_api_key' not in st.session_state: st.session_state.openai_api_key = openai_api_key
if "indexed_urls" not in st.session_state: st.session_state.indexed_urls = set()

# --- 1. URL 去重辅助函数 ---
def get_url_hash(url: str) -> str:
    """生成 URL 的 MD5 哈希值"""
    return hashlib.md5(url.encode()).hexdigest()

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

def get_all_indexed_urls(client: QdrantClient) -> set:
    """从数据库获取所有已索引的 URL（启动时同步用）"""
    # 检查集合是否存在
    if not client.collection_exists(COLLECTION_NAME):
        return set()
    
    urls = set()
    offset = None
    while True:
        result = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True
        )
        points, offset = result
        if not points:
            break
        for point in points:
            if point.payload and "metadata" in point.payload:
                url = point.payload["metadata"].get("source")
                if url:
                    urls.add(url)
        if offset is None:
            break
    return urls

def create_url_index(client: QdrantClient):
    """为 url_hash 创建索引，加速查询"""
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="metadata.url_hash",
            field_schema="keyword"
        )
    except Exception:
        pass  # 索引已存在则忽略

def handle_old_documents(client: QdrantClient, delete: bool = False) -> int:
    """统计或删除没有 url_hash 的旧文档
    Args:
        delete: False=仅统计, True=统计并删除
    Returns:
        旧文档数量
    """
    # 检查集合是否存在
    if not client.collection_exists(COLLECTION_NAME):
        return 0
    
    ids_to_delete = []
    offset = None
    while True:
        result = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True
        )
        points, offset = result
        if not points:
            break
        for point in points:
            if point.payload:
                metadata = point.payload.get("metadata", {})
                if "url_hash" not in metadata:
                    ids_to_delete.append(point.id)
        if offset is None:
            break
    
    if delete and ids_to_delete:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=ids_to_delete
        )
    return len(ids_to_delete)

# --- 2. 资源初始化 ---
@st.cache_resource(show_spinner="正在加载 BGE 模型...")
def get_resources(qdrant_host, qdrant_api_key):
    try:
        model_name = "BAAI/bge-small-zh-v1.5"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

        client = QdrantClient(qdrant_host, api_key=qdrant_api_key)

        if not client.collection_exists(COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
            )

        # 创建 url_hash 索引优化查询
        create_url_index(client)

        db = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embedding_model
        )
        return db, client  # 同时返回 client
    except Exception as e:
        st.error(f"初始化失败: {str(e)}")
        return None, None

# --- 2. LLM ---
def get_llm(api_key, json_mode=False):
    return ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=api_key,
        openai_api_base="https://openai.api2d.net/v1",
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}} if json_mode else {}
    )

# --- 3. 核心节点 ---
def grade_documents(state, api_key) -> Literal["generate", "rewrite"]:
    mode = state.get("run_mode", "deep")
    
    if mode == "fast":
        print("🚀 快速模式")
        return "generate"

    print("---🔎 GRADE ---")
    current_step = state.get("loop_step", 0)

    if current_step >= 3:
        return "generate"

    class Grade(BaseModel):
        binary_score: str = Field(description="'yes' or 'no'")

    model = get_llm(api_key)
    llm_with_tool = model.with_structured_output(Grade)

    # 🔥 关键修改3: 更宽容的评分提示
    prompt = ChatPromptTemplate.from_template(
        """你是评分员。评估检索片段是否能回答问题。
        
        规则：
        1. 如果片段包含问题中提到的概念/术语的定义或解释，评为 'yes'
        2. 即使用词不完全匹配（如"概念"vs"定义"），但语义相关，也评为 'yes'
        3. 只有完全无关时才评为 'no'
        
        问题: {question}
        片段: {context}
        
        评分 (yes/no):"""
    )
    
    messages = state["messages"]
    question = messages[0].content
    docs = messages[-1].content
    
    print(f"👉 问题: {question}")
    print(f"📄 片段: {docs[:100]}...")

    chain = prompt | llm_with_tool
    score = chain.invoke({"question": question, "context": docs}).binary_score

    if score == "yes":
        print("✅ 通过")
        return "generate"
    else:
        print("❌ 重写")
        return "rewrite"

def agent(state, tools, api_key):
    print("---🤖 AGENT ---")
    model = get_llm(api_key).bind_tools(tools)
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def rewrite(state, api_key):
    print("---✍️ REWRITE ---")
    question = state["messages"][0].content
    current_step = state.get("loop_step", 0)
    
    # 🔥 关键修改4: 更智能的重写策略
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
    print(f"📝 重写结果: {rewritten}")
    
    return {
        "messages": [HumanMessage(content=rewritten)], 
        "loop_step": current_step + 1
    }

def generate(state, api_key):
    print("---💡 GENERATE ---")
    messages = state["messages"]
    question = messages[0].content
    
    docs = ""
    for m in reversed(messages):
        if m.type == "tool":
            docs = m.content
            break
    
    if not docs:
        docs = "未检索到相关文档。"

    # 🔥 关键修改5: 更清晰的生成提示
    prompt = ChatPromptTemplate.from_template(
        """你是一个贴心、精准的问答助手。
        
        【检索到的文档片段】:
        {context}
        
        【用户问题】:
        {question}
        
        【要求】:
        1. 如果文档中有完整定义，直接引用
        2. 不要说"根据文档"之类的套话
        3. 语义相关的表述视为匹配（如"概念"="定义"）
        4. 如果你认为文档内容没有清晰回答问题，或者不够详细以及清楚，还是需要根据文档内容进一步分析，保证切实有效解决用户的问题，比如你认为不够清晰，可以详细解释，或进一步阐述
        
        【你的回答】:"""
    )
    
    rag_chain = prompt | get_llm(api_key) | StrOutputParser()
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}

# --- 4. 构建图 ---
def get_graph(retriever_tool, api_key):
    tools = [retriever_tool]
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", partial(agent, tools=tools, api_key=api_key))
    workflow.add_node("retrieve", ToolNode(tools))
    workflow.add_node("rewrite", partial(rewrite, api_key=api_key))
    workflow.add_node("generate", partial(generate, api_key=api_key))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition, {"tools": "retrieve", END: END})
    workflow.add_conditional_edges("retrieve", partial(grade_documents, api_key=api_key))
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")

    return workflow.compile()

# --- 5. 辅助函数 ---
def add_documents_to_db(url, db, client):
    try:
        # 双重检查：先检查数据库是否已存在
        if check_url_exists_in_db(client, url):
            return False, "URL 已存在于数据库中"
        
        url_hash = get_url_hash(url)
        loader = WebBaseLoader(url)
        docs = loader.load()
        
        # 为每个文档添加 url_hash 元数据
        for doc in docs:
            doc.metadata["url_hash"] = url_hash
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )
        doc_chunks = text_splitter.split_documents(docs)
        
        print(f"\n📦 切分了 {len(doc_chunks)} 个片段:")
        for i, chunk in enumerate(doc_chunks[:3]):
            print(f"  [{i}] {chunk.page_content[:100]}...")
        
        uuids = [str(uuid4()) for _ in range(len(doc_chunks))]
        db.add_documents(documents=doc_chunks, ids=uuids)
        return True, len(doc_chunks)
    except Exception as e:
        return False, str(e)

def generate_message(graph, inputs):
    final_ans = ""
    for output in graph.stream(inputs):
        for key, value in output.items():
            if key == "generate":
                msg = value["messages"][0]
                final_ans = msg if isinstance(msg, str) else msg.content
    return final_ans

# --- Main UI ---
def main():
    with st.sidebar:
        st.subheader("⚙️ 系统设置")
        run_mode = st.radio(
            "运行模式:",
            ["🚀 快速直通", "🧠 深度评测"],
            index=0
        )
        mode_code = "fast" if "快速" in run_mode else "deep"
        
        st.divider()
        st.subheader("🔑 API 配置")
        q_host = st.text_input("Qdrant Host", value=st.session_state.qdrant_host, type="password")
        q_key = st.text_input("Qdrant Key", value=st.session_state.qdrant_api_key, type="password")
        oa_key = st.text_input("API2D Key", value=st.session_state.openai_api_key, type="password")
        
        if st.button("💾 保存"):
            st.session_state.qdrant_host = q_host
            st.session_state.qdrant_api_key = q_key
            st.session_state.openai_api_key = oa_key
            st.success("已保存")

    st.title("🔧 修复版中文 RAG")

    if not all([st.session_state.qdrant_host, st.session_state.qdrant_api_key, st.session_state.openai_api_key]):
        st.info("👈 请先配置 API")
        return

    # 获取 db 和 client
    result = get_resources(st.session_state.qdrant_host, st.session_state.qdrant_api_key)
    if result[0] is None:
        return
    db, client = result

    # 🔥 启动时同步：从数据库加载已索引的 URL
    if "urls_synced" not in st.session_state:
        try:
            db_urls = get_all_indexed_urls(client)
            st.session_state.indexed_urls.update(db_urls)
            st.session_state.urls_synced = True
        except Exception as e:
            st.warning(f"同步 URL 失败: {e}")

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 10,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }
    )
    retriever_tool = create_retriever_tool(
        retriever, 
        "retrieve_blog_posts", 
        "搜索博客中的概念定义和解释"
    )

    # 📚 知识库管理
    with st.expander("📚 知识库管理", expanded=True):
        # URL 录入区
        st.markdown("#### 添加新 URL")
        col1, col2 = st.columns([3, 1])
        url = col1.text_input("URL:", label_visibility="collapsed", placeholder="https://...")
        if col2.button("📥 存入"):
            if url:
                # 双重检查：先检查内存缓存
                if url in st.session_state.indexed_urls:
                    st.warning("⚠️ 该 URL 已存在（内存缓存）")
                else:
                    with st.spinner("检查并爬取中..."):
                        success, msg = add_documents_to_db(url, db, client)
                        if success:
                            st.session_state.indexed_urls.add(url)
                            st.success(f"✅ 存入 {msg} 个片段")
                        else:
                            if "已存在" in str(msg):
                                # 数据库中存在但本地缓存没有，同步一下
                                st.session_state.indexed_urls.add(url)
                                st.warning(f"⚠️ {msg}")
                            else:
                                st.error(f"❌ {msg}")
        
        st.divider()
        
        # 已索引 URL 列表（简洁显示）
        url_count = len(st.session_state.indexed_urls)
        if url_count > 0:
            with st.expander(f"📋 已索引 {url_count} 个 URL", expanded=False):
                for url in st.session_state.indexed_urls:
                    st.code(url, language=None)
        else:
            st.info("暂无已索引的 URL")
        
        st.divider()
        
        # 旧数据清理（合并为一个按钮）
        if st.button("🧹 检查并清理旧数据"):
            with st.spinner("处理中..."):
                count = handle_old_documents(client, delete=False)
                if count > 0:
                    deleted = handle_old_documents(client, delete=True)
                    st.success(f"✅ 已清理 {deleted} 个旧文档")
                    st.session_state.urls_synced = False
                    st.rerun()
                else:
                    st.success("✅ 无需清理，所有文档都有 url_hash")

    # 问答区
    if st.session_state.indexed_urls:
        st.divider()
        query = st.text_area("🧠 提问:", height=100, placeholder="例如：同化的概念是什么？")
        
        col1, col2 = st.columns([1, 4])
        run_btn = col1.button("▶️ 运行", use_container_width=True)
        test_btn = col2.button("🔍 测试检索（调试用）", use_container_width=True)
        
        if test_btn and query:
            with st.spinner("检索中..."):
                docs = retriever.invoke(query)
                st.write(f"检索到 {len(docs)} 个片段:")
                for i, doc in enumerate(docs[:5]):
                    st.info(f"**[{i}]** {doc.page_content[:200]}...")
        
        if run_btn and query:
            graph = get_graph(retriever_tool, st.session_state.openai_api_key)
            inputs = {
                "messages": [HumanMessage(content=query)], 
                "loop_step": 0,
                "run_mode": mode_code
            }
            
            with st.spinner(f"运行中 ({mode_code})..."):
                try:
                    ans = generate_message(graph, inputs)
                    if ans:
                        st.markdown("### 📝 回答:")
                        st.success(ans)
                    else:
                        st.error("未能生成回答")
                except Exception as e:
                    st.error(f"错误: {str(e)}")

if __name__ == "__main__":
    main()