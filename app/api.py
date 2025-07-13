from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
import os, json
from dotenv import load_dotenv
from .radio_rag import *
from .detect_question_type import *
# from .agent_tool import *
from .util import *
from .chitchat import chitcat_chat
from .detect_search_web import *
from .detect_retriver import *
from .web_search import *

load_dotenv()

app = FastAPI()

SESSION = {}
CHAT_HISTORY = {} 

@app.post("/webhook")
async def line_webhook(request: Request, background_tasks: BackgroundTasks):
    body = await request.body()
    body_json = json.loads(body.decode("utf-8"))
    events = body_json.get("events", [])
    for event in events:
        if event["type"] == "message" and event["message"]["type"] == "text":
            user_query = event["message"]["text"]
            user_id = event["source"]["userId"]
            background_tasks.add_task(handle_user_message, user_query, user_id)
    return JSONResponse(content={"status": "ok"})


def add_message_to_history(user_id, role, text):
    if user_id not in CHAT_HISTORY:
        CHAT_HISTORY[user_id] = []
    CHAT_HISTORY[user_id].append({"role": role, "text": text})
    print("CHAT_HISTORY:", CHAT_HISTORY)

def get_history(user_id, limit=10):
    """คืน history ย้อนหลัง limit รายการ"""
    return CHAT_HISTORY.get(user_id, [])[-limit:]

def send_and_save(user_id, answer):
    send_line_message(user_id, answer)
    add_message_to_history(user_id, "assistant", answer)

# HAVE CHAT HISTORY 
def handle_user_message(user_query, user_id):
    try:
        add_message_to_history(user_id, "user", user_query)
        # ===== 2. Detect language and question type =====
        lang = detect_lang(user_query)
        print('lang:', lang)
        qt = detect_qt_pipeline(user_query, lang)
        print('qt:', qt)
        
        # ===== 1. Handle web search session (เฉพาะ neu_med_thech_qt เท่านั้น) =====
        if SESSION.get(user_id, {}).get("waiting_for_websearch_answer"):
            # ถ้า last_qt ไม่ใช่ neu_med_thech_qt ให้ข้าม websearch ทันที
            if SESSION[user_id].get("last_qt") != "neu_med_thech_qt":
                # send_line_message(user_id, "ขอโทษค่ะ หนูค้นจากอินเทอร์เน็ตได้เฉพาะคำถามด้านเวชศาสตร์นิวเคลียร์นะคะ 🥲🙏🏻")
                send_and_save(user_id, "ขอโทษค่ะ หนูค้นจากอินเทอร์เน็ตได้เฉพาะคำถามด้านเวชศาสตร์นิวเคลียร์นะคะ 🥲🙏🏻")
                del SESSION[user_id]
                return

            search = classify_search_intent(user_query, llm)
            print('search:', search)
            if search == "YES":
                last_query = SESSION[user_id]["last_query"]
                answer = search_web_rag(last_query, llm)
                del SESSION[user_id]
                # send_line_message(user_id, answer)
                send_and_save(user_id, answer)
            else:
                # send_line_message(user_id, "งั้นไว้คราวหน้านะคะ 😊")
                send_and_save(user_id, "งั้นไว้คราวหน้านะคะ 😊")
                del SESSION[user_id]
            return

        # ===== 3. Nuclear Medicine Technical Question → RAG Agent =====
        if qt == 'neu_med_thech_qt':
            if lang == 'th':
                # send_line_message(user_id, 'รอก่อนนะคะ🥺 หนูขอเวลาคิดแป๊บนึง... 🧠💭⏳')
                send_and_save(user_id, 'รอก่อนนะคะ🥺 หนูขอเวลาคิดแป๊บนึง... 🧠💭⏳')
            elif lang == 'en':
                # send_line_message(user_id, 'Please hold on 🥺 I need a moment to think... 🧠💭⏳')
                send_and_save(user_id, 'Please hold on 🥺 I need a moment to think... 🧠💭⏳')

            # ===== 4. Detect if user directly requests web search =====
            search = classify_search_intent(user_query, llm)
            print('search:', search)
            if search == "YES":
                answer = search_web_rag(user_query, llm)
                # send_line_message(user_id, answer)
                send_and_save(user_id, answer)
                return

            # ===== 5. Select Retriever =====
            history = get_history(user_id, limit=5)
            print('Use-history:', history)
            retriver = detect_retriver(user_query)
            print('Retriver:', retriver)
            if retriver == "HOTLAB":
                answer = rag_hotlab(user_query, history=history)
            elif retriver == "PROTOCOL":
                answer = rag_protocol(user_query, history=history)
            elif retriver == "BMD":
                answer = rag_bmd(user_query, history=history)
            elif retriver == "IODINE":
                answer = rag_iodine(user_query, history=history)
            else:
                answer = "ไม่มีข้อมูลในส่วนนี้ค่ะ ขอโทษด้วยนะคะ🙏🏻😭"
            print('RAG-answer:', answer)

            # ===== 6. Not found in RAG → Ask for web search =====
            if any(kw in answer for kw in ["ไม่มีข้อมูล", "Sorry", "don't have information", "ขอโทษ"]):
                SESSION[user_id] = {
                    "waiting_for_websearch_answer": True,
                    "last_query": user_query,
                    "last_qt": qt
                }
                followup = (
                    "หนูไม่มีข้อมูลในคลัง 😢\n🌐 ให้หนูลองค้นจากอินเทอร์เน็ตไหมคะ?"
                    if lang == 'th' else
                    "I don't have the info 😢\n🌐 Would you like me to search the internet?"
                )
                # send_line_message(user_id, followup)
                send_and_save(user_id, followup)
            else:
                # send_line_message(user_id, answer)
                send_and_save(user_id, answer)

        # ===== 4. Chitchat =====
        elif qt == 'chitchat_qt':
            history = get_history(user_id, limit=5) 
            print('Use-history:', history)
            answer = chitcat_chat(user_query, history)
            # send_line_message(user_id, answer)
            send_and_save(user_id, answer)

        # ===== 5. Other Technical (not nuclear medicine) =====
        elif qt == 'other_thech_qt':
            if lang == 'th':
                # send_line_message(
                #     user_id,
                #     "ไม่สามารถตอบได้ค่ะ เนื่องจากเกินขอบเขตความรับผิดชอบของหนู เพราะคุณถามคำถามที่ไม่เกี่ยวข้องกับเวชศาสตร์นิวเคลียร์ค่ะ หนูขอโทษนะคะ🙏🏻😭 หากมีคำถามเพิ่มเติมเกี่ยวกับการตรวจหรือสารเภสัชรังสี สามารถถามหนูได้เลยนะคะ 🤗"
                # )
                send_and_save(
                    user_id,
                    "ไม่สามารถตอบได้ค่ะ เนื่องจากเกินขอบเขตความรับผิดชอบของหนู เพราะคุณถามคำถามที่ไม่เกี่ยวข้องกับเวชศาสตร์นิวเคลียร์ค่ะ หนูขอโทษนะคะ🙏🏻😭 หากมีคำถามเพิ่มเติมเกี่ยวกับการตรวจหรือสารเภสัชรังสี สามารถถามหนูได้เลยนะคะ 🤗"
                )
            elif lang == 'en':
                # send_line_message(
                #     user_id,
                #     "I’m sorry, I cannot answer this question as it is beyond my area of responsibility. Your question is not related to Nuclear Medicine. 🙏🏻😭 If you have any questions about nuclear medicine procedures or radiopharmaceuticals, feel free to ask me anytime! 🤗"
                # )
                send_and_save(
                    user_id,
                    "I’m sorry, I cannot answer this question as it is beyond my area of responsibility. Your question is not related to Nuclear Medicine. 🙏🏻😭 If you have any questions about nuclear medicine procedures or radiopharmaceuticals, feel free to ask me anytime! 🤗"
                )
            # *** ไม่มี SESSION websearch สำหรับ qt นี้ ***

        # ===== 6. Other/Error =====
        else:
            # send_line_message(user_id, "เกิดข้อผิดพลาด กรุณาลองใหม่🙏🏻" if lang == 'th'
            #                   else "Oops! Something went wrong. Please try again. 🙏🏻")
            send_and_save(user_id, "เกิดข้อผิดพลาด กรุณาลองใหม่🙏🏻" if lang == 'th'
                              else "Oops! Something went wrong. Please try again. 🙏🏻")
    except Exception as e:
        print("ERROR:", e)
        try:
            if lang == 'th':
                # send_line_message(user_id, "เกิดข้อผิดพลาด กรุณาลองใหม่🙏🏻")
                send_and_save(user_id, "เกิดข้อผิดพลาด กรุณาลองใหม่🙏🏻")
            elif lang == 'en':
                # send_line_message(user_id, "Oops! Something went wrong. Please try again. 🙏🏻")
                send_and_save(user_id, "Oops! Something went wrong. Please try again. 🙏🏻")
            else:
                # send_line_message(user_id, "เกิดข้อผิดพลาด กรุณาลองใหม่🙏🏻")
                send_and_save(user_id, "เกิดข้อผิดพลาด กรุณาลองใหม่🙏🏻")

        except Exception:
            pass

# NO CHAT HISTORY 
# def handle_user_message(user_query, user_id):
#     try:
#         # ===== 2. Detect language and question type =====
#         lang = detect_lang(user_query)
#         print('lang:', lang)
#         qt = detect_qt_pipeline(user_query, lang)
#         print('qt:', qt)
        
#         # ===== 1. Handle web search session (เฉพาะ neu_med_thech_qt เท่านั้น) =====
#         if SESSION.get(user_id, {}).get("waiting_for_websearch_answer"):
#             # ถ้า last_qt ไม่ใช่ neu_med_thech_qt ให้ข้าม websearch ทันที
#             if SESSION[user_id].get("last_qt") != "neu_med_thech_qt":
#                 send_line_message(user_id, "ขอโทษค่ะ หนูค้นจากอินเทอร์เน็ตได้เฉพาะคำถามด้านเวชศาสตร์นิวเคลียร์นะคะ 🥲🙏🏻")
#                 del SESSION[user_id]
#                 return

#             search = classify_search_intent(user_query, llm)
#             print('search:', search)
#             if search == "YES":
#                 last_query = SESSION[user_id]["last_query"]
#                 answer = search_web_rag(last_query, llm)
#                 del SESSION[user_id]
#                 send_line_message(user_id, answer)
#             else:
#                 send_line_message(user_id, "งั้นไว้คราวหน้านะคะ 😊")
#                 del SESSION[user_id]
#             return

#         # ===== 3. Nuclear Medicine Technical Question → RAG Agent =====
#         if qt == 'neu_med_thech_qt':
#             if lang == 'th':
#                 send_line_message(user_id, 'รอก่อนนะคะ🥺 หนูขอเวลาคิดแป๊บนึง... 🧠💭⏳')
#             elif lang == 'en':
#                 send_line_message(user_id, 'Please hold on 🥺 I need a moment to think... 🧠💭⏳')

#             # ===== 4. Detect if user directly requests web search =====
#             search = classify_search_intent(user_query, llm)
#             print('search:', search)
#             if search == "YES":
#                 answer = search_web_rag(user_query, llm)
#                 send_line_message(user_id, answer)
#                 return

#             # ===== 5. Select Retriever =====
#             retriver = detect_retriver(user_query)
#             print('Retriver:', retriver)
#             if retriver == "HOTLAB":
#                 answer = rag_hotlab(user_query)
#             elif retriver == "PROTOCOL":
#                 answer = rag_protocol(user_query)
#             elif retriver == "BMD":
#                 answer = rag_bmd(user_query)
#             elif retriver == "IODINE":
#                 answer = rag_iodine(user_query)
#             else:
#                 answer = "ไม่มีข้อมูลในส่วนนี้ค่ะ ขอโทษด้วยนะคะ🙏🏻😭"
#             print('RAG-answer:', answer)

#             # ===== 6. Not found in RAG → Ask for web search =====
#             if any(kw in answer for kw in ["ไม่มีข้อมูล", "Sorry", "don't have information", "ขอโทษ"]):
#                 SESSION[user_id] = {
#                     "waiting_for_websearch_answer": True,
#                     "last_query": user_query,
#                     "last_qt": qt
#                 }
#                 followup = (
#                     "หนูไม่มีข้อมูลในคลัง 😢\n🌐 ให้หนูลองค้นจากอินเทอร์เน็ตไหมคะ?"
#                     if lang == 'th' else
#                     "I don't have the info 😢\n🌐 Would you like me to search the internet?"
#                 )
#                 send_line_message(user_id, followup)
#             else:
#                 send_line_message(user_id, answer)

#         # ===== 4. Chitchat =====
#         elif qt == 'chitchat_qt':
#             answer = chitcat_chat(user_query)
#             send_line_message(user_id, answer)

#         # ===== 5. Other Technical (not nuclear medicine) =====
#         elif qt == 'other_thech_qt':
#             if lang == 'th':
#                 send_line_message(
#                     user_id,
#                     "ไม่สามารถตอบได้ค่ะ เนื่องจากเกินขอบเขตความรับผิดชอบของหนู เพราะคุณถามคำถามที่ไม่เกี่ยวข้องกับเวชศาสตร์นิวเคลียร์ค่ะ หนูขอโทษนะคะ🙏🏻😭 หากมีคำถามเพิ่มเติมเกี่ยวกับการตรวจหรือสารเภสัชรังสี สามารถถามหนูได้เลยนะคะ 🤗"
#                 )
#             elif lang == 'en':
#                 send_line_message(
#                     user_id,
#                     "I’m sorry, I cannot answer this question as it is beyond my area of responsibility. Your question is not related to Nuclear Medicine. 🙏🏻😭 If you have any questions about nuclear medicine procedures or radiopharmaceuticals, feel free to ask me anytime! 🤗"
#                 )
#             # *** ไม่มี SESSION websearch สำหรับ qt นี้ ***

#         # ===== 6. Other/Error =====
#         else:
#             send_line_message(user_id, "เกิดข้อผิดพลาด กรุณาลองใหม่🙏🏻" if lang == 'th'
#                               else "Oops! Something went wrong. Please try again. 🙏🏻")

#     except Exception as e:
#         print("ERROR:", e)
#         try:
#             if lang == 'th':
#                 send_line_message(user_id, "เกิดข้อผิดพลาด กรุณาลองใหม่🙏🏻")
#             elif lang == 'en':
#                 send_line_message(user_id, "Oops! Something went wrong. Please try again. 🙏🏻")
#             else:
#                 send_line_message(user_id, "เกิดข้อผิดพลาด กรุณาลองใหม่🙏🏻")
#         except Exception:
#             pass


@app.get("/webhook")
async def check_webhook():
    return {"status": "webhook is alive"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
