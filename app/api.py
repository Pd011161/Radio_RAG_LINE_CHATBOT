from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
import hmac, hashlib, os, json, requests, pickle, faiss
import numpy as np
from dotenv import load_dotenv
load_dotenv()

# ============ ENV ============
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
openapi_key = os.getenv("OPENAI_API_KEY")

# ============ LOAD DB ============
index = faiss.read_index("database/index.index")
with open("database/id2text.pkl", "rb") as f:
    id2text = pickle.load(f)
with open("database/id2meta.pkl", "rb") as f:
    id2meta = pickle.load(f)

# ============ EMBEDDER ============
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("intfloat/multilingual-e5-base")

# ============ LangChain ============
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()

prompt_template = PromptTemplate.from_template(
    """หนูชื่อ 'น้องนิวเคลียร์ 🤖☢️' หนูเป็นผู้ช่วยให้คำปรึกษาด้านเวชศาสตร์นิวเคลียร์ งานของหนูคือการคำถามด้านเวชศาสตร์นิวเคลียร์จากคำถามโดยใช้ข้อมูลอ้างอิง

[ข้อมูลอ้างอิง]
{context}

[คำถามผู้ใช้]
{question}

[แนวทางการตอบกลับ]
- ให้หนูตอบโดยอ้างอิงข้อมูลจาก [ข้อมูลอ้างอิง] เท่านั้น และขอสรุปหรืออธิบายให้ครบถ้วน ชัดเจน น่ารัก และสุภาพ
- หากคำถามเป็นคำถามเชิงเทคนิคที่ไม่มีคำตอบใน [ข้อมูลอ้างอิง] ให้หนูตอบว่า "ไม่มีข้อมูลในส่วนนี้ค่ะ ขอโทษด้วยนะคะ🙏🏻😭"
- แต่ถ้าคำถามเป็นลักษณะทั่วไป(คำถามสารทุกข์สุกดิบ) (เช่น ทักทาย ขอบคุณ พูดคุยเล่นฝฝ) ที่ไม่ใช่ข้อมูลเชิงเทคนิคหรือความรู้เฉพาะ หนูสามารถตอบได้อย่างสุภาพและน่ารัก
- หนูจะไม่ใช้ข้อมูลอื่นนอกเหนือจาก [ข้อมูลอ้างอิง] ในการตอบ ยกเว้นกรณีเป็นคำถามทั่วไปตามที่ระบุไว้ข้างต้น
- ขอให้แทนตัวเองว่า "หนู" ทุกครั้งในการตอบ และรูปแบบการตอบให้น่ารักและเป็นมิตร สุภาพลงท้ายด้วย 'นะคะ' หรือ ค่ะ' และใ้ช้ emoji เพื่อเพิ่มความเป็นมิตรได้

[รูปแบบการตอบกลับ]
ตอบกลับในรูปแบบ JSON เท่านั้น:
{{
  "answer": "คำตอบของหนูที่นี่"
}}

**หมายเหตุ:** กรุณาตอบเป็น JSON เท่านั้น ห้ามมีข้อความอื่นเพิ่มเติมนอกเหนือจากนี้ค่ะ
"""
)


llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.0,
    max_tokens=512,
    openai_api_key=openapi_key
)

chain = prompt_template | llm | parser

def rag_pipeline(user_query: str, embedder, index, id2text, top_k=3):
    query_embed = embedder.encode([user_query])
    D, I = index.search(np.array(query_embed, dtype=np.float32), top_k)
    top_chunks = [id2text[i] for i in I[0]]
    context = "\n\n".join(top_chunks)
    inputs = {"context": context, "question": user_query}
    result = chain.invoke(inputs)
    return result["answer"], context, top_chunks

# ============ FASTAPI ============
app = FastAPI()

@app.post("/webhook")
async def line_webhook(request: Request, background_tasks: BackgroundTasks):
    body = await request.body()
    body_json = json.loads(body.decode("utf-8"))
    events = body_json.get("events", [])
    for event in events:
        if event["type"] == "message" and event["message"]["type"] == "text":
            user_query = event["message"]["text"]
            user_id = event["source"]["userId"]
            # ทำใน background เพื่อไม่ timeout!
            background_tasks.add_task(handle_user_message, user_query, user_id)
    return JSONResponse(content={"status": "ok"})



def handle_user_message(user_query, user_id):
    try:
        # 1. ส่ง effect ก่อน (แจ้ง user ว่ารอก่อนนะ)
        send_line_message(user_id, 'รอก่อนนะคะ🥺 หนูขอเวลาคิดแป๊บนึง... 🧠💭⏳')
        
        # 2. ประมวลผลจริง
        answer, context, chunks = rag_pipeline(user_query, embedder, index, id2text)
        
        # 3. ส่งคำตอบจริงอีกที
        send_line_message(user_id, answer)
    except Exception as e:
        print("ERROR:", e)
        send_line_message(user_id, "เกิดข้อผิดพลาด กรุณาลองใหม่")


def send_line_message(user_id, text):
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
    }
    data = {
        "to": user_id,
        "messages": [{"type": "text", "text": text}],
    }
    resp = requests.post(url, headers=headers, json=data)
    print("LINE response:", resp.status_code, resp.text)

# สำหรับเทส GET (optional)
@app.get("/webhook")
async def check_webhook():
    return {"status": "webhook is alive"}

# ============ START ============
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
