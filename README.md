# 🚀 RAG Nuclear Medicine Chatbot (LINE RAG Bot)

An AI-based Retrieval-Augmented Generation (RAG) chatbot for answering nuclear medicine questions in Thai (and English), integrated with LINE messaging, powered by FastAPI, FAISS, and OpenAI/LangChain.

---

## 📚 Project Description

“Nong Nuclear 🤖☢️” is a LINE chatbot designed to answer and assist with nuclear medicine questions.  
It uses Retrieval-Augmented Generation (RAG): retrieves context from a document knowledge base via FAISS + Embeddings and generates answers with an LLM.  
Integrates with the LINE Messaging API via webhook and is built using FastAPI (Python).

---

## ✨ Features

- Thai and English language support
- Answers technical questions only from the knowledge base (if no answer found: replies politely with "No information available")
- Can handle small talk and general greetings
- LINE Bot integration (via webhook, push message to users)
- Returns answers in JSON format (easy for further integration)

---

## 🛠️ Tech Stack

- **Backend:** Python, FastAPI, Uvicorn
- **NLP:** sentence-transformers, LangChain, OpenAI GPT-4o (or gpt-4o-mini)
- **Vector DB:** FAISS
- **LINE Messaging API**
- **Ngrok:** for temporary public webhook access
- **Docker:** for easy deployment

---

## 🚦 Getting Started

### 1. Clone & Install

```bash
git clone <repo-url>
cd RAG
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
RAG/
├── app/
│   └── api.py
├── data/
│   └── hotlab-v1.txt
├── database/
│   ├── id2meta.pkl
│   ├── id2text.pkl
│   └── index.index
├── test/
│   └── prepare-rag.ipynb
├── .env
├── .gitignore
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── README.md
```

---

## 🧑‍💻 Example Usage

- Users send messages to the LINE Bot
- The bot responds with referenced information from the knowledge base (if not found, replies “No information available, sorry 🙏🏻😭”)
- You can use the public ngrok URL as a webhook for LINE

---

## 📝 Notes

- Do not commit .env, .pkl files, or other secrets to Git
- Check .gitignore for ignored files/folders (recommended: ignore large database/data artifacts)
- Example embedding model: intfloat/multilingual-e5-base
- To add more documents, update the FAISS index

---

## ⚡ Quick Summary

- Python RAG chatbot with FAISS + LangChain + LINE Webhook
- Great for nuclear medicine domain QA
- Deployable both locally and with Docker



