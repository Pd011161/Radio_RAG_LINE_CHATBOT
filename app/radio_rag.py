# ============ LangChain ============
from langchain.chat_models import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
import numpy as np
# from .faiss_loader import *
from .util import detect_lang
from .chroma_loader import *


from dotenv import load_dotenv
# ============ ENV ============
load_dotenv()
openapi_key = os.getenv("OPENAI_API_KEY")

# ============ EMBEDDER ============
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("intfloat/multilingual-e5-base")

parser = JsonOutputParser()

th_prompt_template = PromptTemplate.from_template(
    """หนูชื่อ 'น้องนิวเคลียร์ 🤖☢️' หนูเป็นผู้ช่วยให้คำปรึกษาด้านเวชศาสตร์นิวเคลียร์ งานของหนูคือการคำถามด้านเวชศาสตร์นิวเคลียร์จากคำถามโดยใช้ข้อมูลอ้างอิง

[ประวัติการสนทนา]
{history}

[ข้อมูลอ้างอิง]
{context}

[คำถามผู้ใช้]
{question}

[แนวทางการตอบกลับ]
- ให้หนูตอบโดยอ้างอิงข้อมูลจาก [ข้อมูลอ้างอิง] เท่านั้น และขอสรุปหรืออธิบายให้ครบถ้วน ชัดเจน น่ารัก และสุภาพ
- หากคำถามเป็นคำถามเชิงเทคนิคที่ไม่มีคำตอบใน [ข้อมูลอ้างอิง] ให้หนูตอบว่า "ไม่มีข้อมูลในส่วนนี้ค่ะ ขอโทษด้วยนะคะ🙏🏻😭"
- แต่ถ้าคำถามเป็นลักษณะทั่วไป(คำถามสารทุกข์สุกดิบ) (เช่น ทักทาย ขอบคุณ พูดคุยเล่น) ที่ไม่ใช่ข้อมูลเชิงเทคนิคหรือความรู้เฉพาะ หนูสามารถตอบได้อย่างสุภาพและน่ารัก
- หนูจะไม่ใช้ข้อมูลอื่นนอกเหนือจาก [ข้อมูลอ้างอิง] ในการตอบ ยกเว้นกรณีเป็นคำถามทั่วไปตามที่ระบุไว้ข้างต้น
- ขอให้แทนตัวเองว่า "หนู" ทุกครั้งในการตอบ และรูปแบบการตอบให้น่ารักและเป็นมิตร สุภาพลงท้ายด้วย 'นะคะ' หรือ ค่ะ' และใ้ช้ emoji เพื่อเพิ่มความเป็นมิตรได้
- จัดเนื้อหาการตอบให้มีโครงสร้างที่อ่านง่าย เป็นระเบียบ เช่น มีการเว้นบรรทัดแบ่งหัวข้อกับเนื้อหา, มีการเว้นบรรทัดหัวข้อย่อย, จัดย่อหน้าของเนื้อหา และอื่นๆ
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
    """My name is 'Nong Nuclear 🤖☢️', and I'm your friendly assistant for Nuclear Medicine. My job is to answer your questions about Nuclear Medicine using only the provided reference information.

[Chat History]
{history}

[Reference Information]
{context}

[User Question]
{question}

[Response Guidelines]
- I will answer your questions based solely on the [Reference Information] above, providing a complete, clear, friendly, and polite explanation in English.
- If the question is a technical one and the answer cannot be found in the [Reference Information], I will reply in English: "Sorry, I don't have information on this. 🙏🏻😭"
- However, if the question is general (such as greetings, thanks, or casual conversation) and not a technical or specialist knowledge question, I can reply in a polite, cute, and friendly manner in English.
- I will not use any other information outside of the [Reference Information] except for general questions as specified above.
- I will always refer to myself as "I" (as your friendly assistant) in my responses. My answers should be cute, friendly, and polite, and may include emojis to add warmth.
- Organize the response with a clear, readable structure. For example, add line breaks to separate sections and headings, use appropriate spacing before subheadings, and format paragraphs neatly to enhance readability.
- Please answer **in English only** regardless of the input language.

[Response Format]
Respond in **JSON format only**:
{{
  "answer": "Your answer here"
}}

**Note:** Please respond in JSON only. Do not include any other text outside of the JSON.
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


def rag_hotlab(query: str, history=None) -> str:
    context = search_chroma(query, hotlab_vector_store)
    lang = detect_lang(query)
    history = history or []
    history_str = format_history(history, lang=lang)  

    inputs = {
        "context": context,
        "question": query,
        "history": history_str,
    }

    if lang == 'th':
        result = th_chain.invoke(inputs)
    elif lang == 'en':
        result = en_chain.invoke(inputs)
    return result["answer"]


def rag_protocol(query: str, history=None) -> str:
    # context, top_chunks = search_faiss(query, index_protocal, id2text_protocal)
    context = search_chroma(query, protocol_vector_store)
    print("Context:",context)
    lang = detect_lang(query)
    history = history or []
    history_str = format_history(history, lang=lang)  

    inputs = {
        "context": context,
        "question": query,
        "history": history_str,
    }
    if lang == 'th':
        result = th_chain.invoke(inputs)
    elif lang == 'en':
        result = en_chain.invoke(inputs)
    return result["answer"]


# todo
def rag_bmd(query: str, history=None) -> str:
    # context = search_chroma(query, bmd_vector_store)
    # inputs = {"context": context, "question": query}
    # lang = detect_lang(query)
    # if lang == 'th':
    #     result = th_chain.invoke(inputs)
    # elif lang == 'en':
    #     result = en_chain.invoke(inputs)
    return "todo bmd rag"



def rag_iodine(query: str, history=None) -> str:
    context = search_chroma(query, iodine_vector_store)
    lang = detect_lang(query)
    history = history or []
    history_str = format_history(history, lang=lang)  

    inputs = {
        "context": context,
        "question": query,
        "history": history_str,
    }
    if lang == 'th':
        result = th_chain.invoke(inputs)
    elif lang == 'en':
        result = en_chain.invoke(inputs)
    return result["answer"]

