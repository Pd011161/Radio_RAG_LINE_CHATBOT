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


def classify_search_intent(user_reply, llm):
    lang = detect_lang(user_reply)
    if lang == 'th':
        prompt = PromptTemplate.from_template('''คำถามผู้ใช้: "{user_reply}"
คำสั่ง: โปรดวิเคราะห์และตัดสินใจว่า ข้อความข้างต้นเป็น "การยินยอมให้ค้นหาข้อมูลเพิ่มเติมทางอินเทอร์เน็ตหรือไม่"

รูปแบบการตอบกลับ: ตอบกลับในรูปแบบ JSON เท่านั้น เช่น
{{
  "permission": "YES"      // หากผู้ใช้ยินยอมให้ค้นหา
}}
หรือ
{{
  "permission": "NO"  // หากผู้ใช้ไม่ยินยอมให้ค้นหา
}}

**หมายเหตุ:** ห้ามมีข้อความอื่นใดนอกเหนือจาก JSON ที่กำหนด
''')
    else:
        prompt = PromptTemplate.from_template('''User message: "{user_reply}"
Instruction: Please analyze and determine whether the above message indicates the user's consent to perform an additional web search.

Response format: Respond in JSON only, for example:
{{
  "permission": "YES"      // If the user gives consent for web search
}}
or
{{
  "permission": "NO"       // If the user does NOT give consent
}}

**Note:** Do not include any text or explanation other than the specified JSON format.
''')

    chain = prompt | llm | parser
    result = chain.invoke({"user_reply": user_reply})
    # result is a dict like {"permission": "YES"}
    return result["permission"].upper()
