from langchain.chat_models import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent
from .radio_rag import *
from .web_search import search_web_rag

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)
SESSION = {}

tools = [
    Tool(
        name="RAG_HOTLAB",
        func=rag_hotlab,
        description="Use this tool to retrieve information about radiopharmaceutical preparation or Hot lab procedures in nuclear medicine."
    ),
    Tool(
        name="RAG_PROTOCAL",
        func=rag_protocol,
        description="Use this tool to retrieve information about clinical protocols and procedures in nuclear medicine examinations."
    ),
    Tool(
        name="RAG_WEB_SEARCH",
        func=search_web_rag,
        description="Use this tool to search the web for information that cannot be found in the knowledge base."
    ),
]


agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="openai-functions",
    verbose=True
)
