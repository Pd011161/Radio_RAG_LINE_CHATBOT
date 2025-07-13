# ============ LangChain ============
from langchain.chat_models import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import  os
# from langchain.tools import DuckDuckGoSearchRun
from ddgs import DDGS
from .radio_rag import *
from .util import detect_lang
from dotenv import load_dotenv

# ============ ENV ============
load_dotenv()
openapi_key = os.getenv("OPENAI_API_KEY")

parser = JsonOutputParser()

# ====== DuckDuckGo Web Search Tool ======
# search_tool_raw = DDGS()
def duckduckgo_search(query, max_results=5):
    with DDGS() as ddgs:
        results = []
        for r in ddgs.text(query):
            results.append(r['body'])
            if len(results) >= max_results:
                break
        return "\n\n".join(results)


llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.0,
    # max_tokens=512,
    openai_api_key=openapi_key
)

# ====== Web Search + Summarization ======
def search_web_rag(query, llm):
    lang = detect_lang(query)
    print('lang:', lang)
    try:
        # web_result = search_tool_raw.run(query)
        web_result = duckduckgo_search(query)

        if lang == 'th':
            prompt = PromptTemplate.from_template(
                """
[ข้อมูลจากเว็บ]
{web_result}

คำถาม: {query}

โปรดตอบคำถามโดยใช้ข้อมูลจากด้านบนเท่านั้น
- ตอบให้น่ารัก สุภาพ และเข้าใจง่าย
- แทนตัวเองว่า "หนู"
- ใช้คำลงท้ายอย่าง "นะคะ", "ค่ะ" และเพิ่ม emoji ได้หากเหมาะสม
- ตอบเป็นภาษาไทยเท่านั้น

ตอบกลับในรูปแบบ JSON เท่านั้น เช่น
{{"answer": "..." }}
"""
            )
        else:
            prompt = PromptTemplate.from_template(
                """
[Web Information]
{web_result}

User question: {query}

Please answer the question using only the information above.
- Respond in English only
- Be polite, friendly, and clear
- Refer to yourself as "I"
- Add emojis if appropriate

Reply in JSON only, for example:
{{"answer": "..." }}
"""
            )
        chain = prompt | llm | parser
        result = chain.invoke({"web_result": web_result, "query": query})
        # result = {"answer": "..."}
        return result["answer"]
    except Exception as e:
        print("[Web Search Error]", e)
        return "ขอโทษค่ะ หนูค้นจากเว็บไม่สำเร็จ ลองใหม่อีกครั้งได้ไหมคะ 🥺🙏🏻" if lang == 'th' else "Sorry, I couldn't search the web properly. Please try again later 🙏🏻"