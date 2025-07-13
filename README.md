# ☢️🤖 Nuclear Medicine RAG Chatbot(LINE)

🎬 Demo Video

[![Watch the demo](https://img.youtube.com/vi/86ZSeTXFCJk/hqdefault.jpg)](https://youtu.be/86ZSeTXFCJk)

> **Nong Nuclear 🤖☢️**  
> Thai/English RAG chatbot for nuclear medicine, integrated with LINE, powered by FastAPI, Chroma, LangChain, OpenAI, and DuckDuckGo Web Search.
> The domain knowledge used in this chatbot is **specific to Nuclear Medicine workflows and practices at Maharat Nakhon Ratchasima Hospital, Thailand**.  
> Answers to technical queries are based on real clinical guidelines, protocols, and common questions encountered in this setting.

---

## 📚 Overview

**Radio_RAG_LINE_CHATBOT** is a bilingual (Thai/English) chatbot for nuclear medicine Q&A on LINE.  
It uses Retrieval-Augmented Generation (RAG) with a Chroma vector database for knowledge retrieval and an LLM (OpenAI GPT-4o or HuggingFace) for answer generation.  
If no answer is found, it can **fallback to DuckDuckGo web search** for up-to-date information.  
Handles both technical queries and small talk, with full LINE integration and context-aware chat history.

---

## ✨ Key Features

- **Bilingual Support:** Thai & English
- **RAG Pipeline:** Retrieves document context from Chroma + generates answers with LLM
- **DuckDuckGo Web Search:** Fallback to web search for missing or out-of-knowledge questions
- **LINE Messaging API:** Seamless LINE chat integration (webhook/push)
- **Chat History:** Context-aware answers using recent chat turns
- **Small Talk & Technical QA:** Friendly chit-chat or advanced nuclear medicine queries
- **JSON Output:** Answers in easy-to-integrate JSON format

---

## 🛠️ Tech Stack

- **Backend:** Python, FastAPI, Uvicorn
- **NLP/RAG:** LangChain, sentence-transformers, OpenAI GPT-4o (or gpt-4o-mini)
- **Vector DB:** Chroma
- **Web Search:** DuckDuckGo API (via LangChain tool)
- **Messaging:** LINE Messaging API
- **Deployment:** Docker, Ngrok (for development)
- **Config:** .env (environment variables)

---

## 🚦 Getting Started

### 1. Clone & Install

```bash
# Clone this repository
git clone https://github.com/Pd011161/Radio_RAG_LINE_CHATBOT.git

# Move into the project directory
cd Radio_RAG_LINE_CHATBOT
```

### 2. Prepare .env file

```bash
LINE_CHANNEL_SECRET=xxxxxxxxxxxxxxxxxxx
LINE_CHANNEL_ACCESS_TOKEN=xxxxxxxxxxxxxxxxxxx
OPENAI_API_KEY=xxxxxxxxxxxxxxxxxxx
NGROK_AUTHTOKEN=xxxxxxxxxxxxxxxxxxx
```

### 3. Run in Docker

```bash
docker build -t radio-rag-line-bot .
docker run -p 8000:8000 --env-file .env radio-rag-line-bot
```

### 4. Get public ngrok URL from logs (to set as LINE webhook)

- You will see a URL like: https://xxxxxxxx.ngrok-free.app/webhook
- Go to LINE Developers Console → Webhook URL → paste this URL

---

## 🧩 Project Structure

```bash
Radio_RAG_LINE_CHATBOT/
├── app/                # FastAPI backend and core logic
├── data/               # Knowledge base text files
├── database/           # Chroma persistent data (vector index, metadata, etc.)
├── test/               # Notebooks and testing scripts
├── requirements.txt / pyproject.toml
├── Dockerfile
├── .env
├── README.md
└── ...
```

---

## 🧑‍💻 Example Usage

- Users chat with the LINE Bot as usual
- The bot tries to answer using the Chroma knowledge base (RAG)
- If no relevant info is found, it automatically searches DuckDuckGo and answers from web content
- Chat history is used for more natural conversations

---

## 📝 Notes

- Do NOT commit: .env, large Chroma DB folders, or secret files to git
- To add new documents: Rebuild or update the Chroma vector database
- Embeddings: Uses intfloat/multilingual-e5-base (or other supported models)
- Docker recommended: for consistency and easy deployment

---

## ⚡ Quick Summary

- RAG chatbot for nuclear medicine Q&A (Thai & English)
- Uses Chroma as the vector database
- DuckDuckGo web search for fallback answers
- FastAPI + Chroma + LangChain + OpenAI + LINE Messaging
- **Knowledge base is tailored for Nuclear Medicine practice at Maharat Nakhon Ratchasima Hospital, Thailand**
- Deploy locally or with Docker in minutes



