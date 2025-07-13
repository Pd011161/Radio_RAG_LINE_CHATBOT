# ============ LangChain ============
from langchain.chat_models import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import hmac, hashlib, os, json, requests, pickle, faiss
import numpy as np
import re
from dotenv import load_dotenv
load_dotenv()

# ============ ENV ============
openapi_key = os.getenv("OPENAI_API_KEY")

parser = JsonOutputParser()

category_prompt_template_th = PromptTemplate.from_template(
"""
คุณเป็นผู้ช่วยในการจำแนกประเภทคำถามของผู้ใช้ ออกเป็น 3 ประเภท

1. **'neu_med_thech_qt'**: คำถามเชิงเทคนิคหรือวิชาชีพที่เกี่ยวกับเวชศาสตร์นิวเคลียร์โดยเฉพาะ เช่น ขั้นตอน, อุปกรณ์, radioisotope, protocol, guideline, การเตรียมสารเภสัชรังสี หรือ Hot lab, การให้น้ำแร่หรือการกลืนแร่ Iodine, การตรวจมวลกระดูก หรือ การหาความหนาแน่นของกระดูก หรือ BMD, หรือหลักการวิทยาศาสตร์ในเวชศาสตร์นิวเคลียร์
2. **'other_thech_qt'**: คำถามเชิงเทคนิคหรือวิชาชีพอื่นที่ไม่ใช่เวชศาสตร์นิวเคลียร์ เช่น ฟิสิกส์, วิทยาศาสตร์ทั่วไป, เทคโนโลยี, หรือคำถามเชิงวิชาการด้านอื่น
3. **'chitchat_qt'**: คำถามทั่วไป ทักทาย พูดคุย สารทุกข์สุกดิบ ขอบคุณ หรือคำถามที่ไม่เกี่ยวกับเนื้อหาทางเทคนิค

[ตัวอย่าง]

เวชศาสตร์นิวเคลียร์ (neu_med_thech_qt):
- "ข้อบ่งชี้ในการใช้ PET/CT คืออะไร"
- "เทคนิคการเตรียมสารเภสัชรังสีคืออะไร"
- "อุปกรณ์ที่ใช้ในห้อง hot lab มีอะไรบ้าง"

เทคนิคอื่น (other_thech_qt):
- "กลไกการออกฤทธิ์ของ aspirin คืออะไร"
- "ฟิสิกส์ของ CT แตกต่างจาก MRI อย่างไร"
- "เทคโนโลยี AI มีประโยชน์อย่างไรในโรงพยาบาล"

ทั่วไป (chitchat_qt):
- "สวัสดีค่ะ"
- "ขอบคุณมากนะคะ"
- "วันนี้อากาศดีจัง"

---

[คำถามผู้ใช้]
{question}

---

**คำสั่ง:**  
จำแนกคำถามข้างต้นให้เป็นแค่หนึ่งใน 3 หมวดนี้  
- 'neu_med_thech_qt'  
- 'other_thech_qt'  
- 'chitchat_qt'  

**ตอบกลับในรูปแบบ JSON เท่านั้น เช่น:**
{{
  "category": "<your_category_here>"
}}

**หมายเหตุ:** ห้ามอธิบายเหตุผลเพิ่มเติม ให้ตอบเฉพาะ JSON เท่านั้น
"""
)


category_prompt_template_en = PromptTemplate.from_template(
"""
You are an assistant that classifies user questions into 3 categories:

1. **'neu_med_thech_qt'**: The question is a technical or professional question specifically about Nuclear Medicine, such as procedures, equipment, radioisotopes, protocols, guidelines, or scientific principles in nuclear medicine.
2. **'other_thech_qt'**: The question is a technical or professional question about other fields, NOT about Nuclear Medicine, such as general science, physics, technology, or academic questions unrelated to nuclear medicine.
3. **'chitchat_qt'**: The question is a general or casual question, such as greetings, thanks, jokes, daily conversation, or any question that is not technical.

[Examples]

Nuclear Medicine Technical (neu_med_thech_qt):
- "What are the indications for PET/CT?"
- "How do you prepare a radiopharmaceutical?"
- "What equipment is used in the hot lab?"

Other Technical (other_thech_qt):
- "What is the mechanism of action of aspirin?"
- "How does CT physics differ from MRI?"
- "What are the benefits of AI technology in hospitals?"

Chitchat (chitchat_qt):
- "Hello!"
- "Thank you very much."
- "The weather is nice today."

---

[User Question]
{question}

---

**Instruction:**  
Classify the user question above into ONLY ONE of the three categories:  
- 'neu_med_thech_qt'  
- 'other_thech_qt'  
- 'chitchat_qt'  

**Return the answer in JSON format only, like this:**
{{
  "category": "<your_category_here>"
}}

**Note:** Only return JSON. Do not explain your reasoning.
"""
)


llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.0,
    # max_tokens=512,
    openai_api_key=openapi_key
)

th_chain = category_prompt_template_th | llm | parser
en_chain = category_prompt_template_en | llm | parser

# def detect_lang(user_query):
    
#     th_count = len(re.findall(r'[ก-๙]', user_query))
#     print('th_count:', th_count)
#     en_count = len(re.findall(r'[a-zA-Z]', user_query))
#     print('en_count:', en_count)

#     if th_count > en_count:
#         lang = 'th'
#         return lang
#     elif th_count < en_count:
#         lang = 'en'
#         return lang 
#     else:  
#         lang = 'th'
#         return lang
    


def detect_lang(user_query):
    th_count = len(re.findall(r'[ก-๙]', user_query))
    en_count = len(re.findall(r'[a-zA-Z]', user_query))
    print('th_count:', th_count)
    print('en_count:', en_count)

    # กรณีพิเศษ ไม่มีตัวอักษรไทยและอังกฤษเลย
    if th_count == 0 and en_count == 0:
        return 'unknown'

    # ถ้าไทยมากกว่าอังกฤษ 80%
    if th_count > en_count * 0.8:
        return 'th'
    # ถ้าอังกฤษมากกว่าไทย 80%
    elif en_count > th_count * 0.8:
        return 'en'
    else:
        return 'th'


def detect_qt_pipeline(user_query: str, lang):
    inputs = {"question": user_query}
    if lang == 'th':
        result = th_chain.invoke(inputs)
    elif lang == 'en':
        result = en_chain.invoke(inputs)
    return result["category"]
