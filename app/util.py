import os
import hmac, hashlib, os, json, requests, pickle, faiss, re
from dotenv import load_dotenv
load_dotenv()

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")


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


def detect_lang(user_query):
    
    th_count = len(re.findall(r'[ก-๙]', user_query))
    en_count = len(re.findall(r'[a-zA-Z]', user_query))

    if th_count > 0:
        lang = 'th'
        return lang
    
    elif th_count == 0 and en_count > 0:
        lang = 'en'
        return lang 
    
    else:  
        lang = 'th'
        return lang



