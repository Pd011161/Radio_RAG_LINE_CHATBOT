# ============ LangChain ============
from langchain.chat_models import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import  os
from .radio_rag import *
from .util import detect_lang
from dotenv import load_dotenv

# ============ ENV ============
load_dotenv()
openapi_key = os.getenv("OPENAI_API_KEY")

parser = JsonOutputParser()


llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.0,
    # max_tokens=512,
    openai_api_key=openapi_key
)


def detect_retriver(user_reply):
    lang = detect_lang(user_reply)
    if lang == 'th':
        prompt = PromptTemplate.from_template(
"""
คุณเป็นผู้ช่วยในการจำแนกประเภทหัวข้อของคำถามของผู้ใช้ ออกเป็น 4 ประเภท

1. **'HOTLAB'**: คำถามเกี่ยวกับการเตรียมสารเภสัชรังสี หรือ Hot lab
2. **'PROTOCOL'**: คำถามที่เกี่ยวกับระเบียบวิธีการในการตรวจทางเวชศาสตร์นิวเคลียร์ หรือ Protocol
3. **'BMD'**: คำถามที่เกี่ยวกับการหาความหนาแน่น หรือมวลของกระดูก หรือ BMD
3. **'IODINE'**: คำถามเกี่ยวกับการให้น้ำแร่ผู้ป่วย หรือการให้กลืนแร่ Iodine หรือ IODINE

[ตัวอย่าง]

การเตรียมสารเภสัชรังสี หรือ Hot lab (HOTLAB):
- "เทคนิคการเตรียมสารเภสัชรังสีคืออะไร"
- "ยาที่ใช้ในการตรวจ"
- "การคำนวนยา"

ระเบียบวิธีการในการตรวจทางเวชศาสตร์นิวเคลียร์ หรือ Protocol ในการตรวจทางเวชศาสตร์นิวเคลียร์ (PROTOCOL):
- "โปรโตคอลของการตรวจ"
- "ระเบียบการตรวจ"
- "วิธีการตรวจ"

การตรวจมวลกระดูก หรือ การหาความหนาแน่นของกระดูก หรือ BMD (BMD):
- "การสแกนความหน้าแน่นของกระดูก"
- "HIP SPINE FEMUR WHOLEBODY"
- "ค่า Z score"
- "ค่า T score"

การให้น้ำแร่ผู้ป่วย หรือการให้กลืนแร่ Iodine 131 หรือ Iodine (IODINE):
- "การให้น้ำแร่ low dose "
- "โลวโดส ไฮโดส "
- "กลืนแร่ "

---

[คำถามผู้ใช้]
{user_reply}

---

**คำสั่ง:**  
จำแนกคำถามข้างต้นให้เป็นแค่หนึ่งใน 4 หมวดนี้  
- 'HOTLAB'  
- 'PROTOCOL'  
- 'BMD'  
- 'IODINE'  

**ตอบกลับในรูปแบบ JSON เท่านั้น เช่น:**
{{
  "category": "<your_category_here>"
}}

**หมายเหตุ:** ห้ามอธิบายเหตุผลเพิ่มเติม ให้ตอบเฉพาะ JSON เท่านั้น
"""
)

    else:
        prompt = PromptTemplate.from_template(
"""
You are an assistant responsible for classifying user questions into one of the following 4 categories:

1. **'HOTLAB'**: Questions related to the preparation of radiopharmaceuticals or Hot lab.
2. **'PROTOCOL'**: Questions about nuclear medicine examination protocols or procedures.
3. **'BMD'**: Questions about bone mineral density measurement or bone mass (BMD).
4. **'IODINE'**: Questions about radioiodine therapy or oral administration of Iodine.

[Examples]

Preparation of radiopharmaceuticals or Hot lab (HOTLAB):
- "What is the technique for preparing radiopharmaceuticals?"
- "What drugs are used for the scan?"
- "How do you calculate the dose?"

Protocols for nuclear medicine examinations (PROTOCOL):
- "What is the protocol for this scan?"
- "What are the procedures for this test?"
- "How is the examination performed?"

Bone mineral density or BMD (BMD):
- "How is bone density scanning done?"
- "HIP SPINE FEMUR WHOLEBODY"
- "What is the Z score?"
- "What is the T score?"

Radioiodine therapy or oral administration of Iodine (IODINE):
- "How is low dose iodine therapy given?"
- "What is the difference between low dose and high dose?"
- "How to administer oral iodine?"

---

[User Question]
{user_reply}

---

**Instruction:**  
Classify the above question into only one of these 4 categories:
- 'HOTLAB'  
- 'PROTOCOL'  
- 'BMD'  
- 'IODINE'  

**Respond in JSON format only, for example:**
{{
  "category": "<your_category_here>"
}}

**Note:** Do not provide any explanation or additional information. Respond with JSON only.
"""
)


    chain = prompt | llm | parser
    result = chain.invoke({"user_reply": user_reply})
    # result is a dict like {"permission": "YES"}
    return result["category"]
