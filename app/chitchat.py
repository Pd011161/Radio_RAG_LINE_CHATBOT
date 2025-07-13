# ============ LangChain ============
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
import numpy as np
# from .faiss_loader import *
from .util import detect_lang
from dotenv import load_dotenv

# ============ ENV ============
load_dotenv()
openapi_key = os.getenv("OPENAI_API_KEY")

# ============ EMBEDDER ============
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("intfloat/multilingual-e5-base")

parser = JsonOutputParser()

th_prompt_template = PromptTemplate.from_template(
    """หนูชื่อ 'น้องนิวเคลียร์ 🤖☢️' หนูเป็นผู้ช่วยให้คำปรึกษาด้านเวชศาสตร์นิวเคลียร์ งานของหนูคือการคำถามด้านเวชศาสตร์นิวเคลียร์จากคำถาม หรือพูดคุยเล่นกับผู้ใช้

[ประวัติการสนทนา]
{history}

[คำถามผู้ใช้]
{question}

[แนวทางการตอบกลับ]
- ให้หนูตอบอย่างชัดเจน น่ารัก และสุภาพ
- ขอให้แทนตัวเองว่า "หนู" ทุกครั้งในการตอบ และรูปแบบการตอบให้น่ารักและเป็นมิตร สุภาพลงท้ายด้วย 'นะคะ' หรือ ค่ะ' และใ้ช้ emoji เพื่อเพิ่มความเป็นมิตรได้
- กรุณาตอบ **เป็นภาษาไทยเท่านั้น** ไม่ว่าจะป้อนภาษาอะไรก็ตาม

[รูปแบบการตอบกลับ]
ตอบกลับในรูปแบบ JSON เท่านั้น:
{{
  "answer": "คำตอบของหนูที่นี่"
}}

**หมายเหตุ:** กรุณาตอบเป็น JSON เท่านั้น ห้ามมีข้อความอื่นเพิ่มเติมนอกเหนือจากนี้ค่ะ
"""
)

eng_prompt_template = PromptTemplate.from_template(
    """My name is 'Nong Nuclear 🤖☢️', and I am your friendly assistant for Nuclear Medicine. My job is to answer your questions about nuclear medicine, or just to have a casual conversation with you.

[Chat History]
{history}

[User Question]
{question}

[Response Guidelines]
- Please answer in a clear, friendly, and polite manner.
- Always refer to yourself as "I" in every response. Make your answers cute, approachable, and end with soft expressions like "na ka" or "ka" to maintain a gentle tone. Feel free to add emojis to make the reply more friendly.
- Please respond **in English only**, no matter what language the question is in.

[Response Format]
Respond in **JSON** format only:
{{
  "answer": "Your answer here"
}}

**Note:** Please reply with only the JSON object above. Do not include any other text outside the JSON format.
"""
)

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.0,
    # max_tokens=512,
    openai_api_key=openapi_key
)

th_chain = th_prompt_template | llm | parser
en_chain = eng_prompt_template | llm | parser


def format_history(history, lang):
    text = ""
    for turn in history:
        if turn["role"] == "user":
            if lang == "th":
                text += "👤 ผู้ใช้: " + turn["text"] + "\n"
            else:
                text += "👤 User: " + turn["text"] + "\n"
        else:
            if lang == "th":
                text += "🤖 หนู: " + turn["text"] + "\n"
            else:
                text += "🤖 Assistant: " + turn["text"] + "\n"
    return text.strip()


def chitcat_chat(query: str, history=None) -> str:
    lang = detect_lang(query)
    history = history or []
    history_str = format_history(history, lang=lang)
    inputs = {
        "question": query,
        "history": history_str,
    }
    if lang == 'th':
        result = th_chain.invoke(inputs)
    elif lang == 'en':
        result = en_chain.invoke(inputs)
    return result["answer"]

