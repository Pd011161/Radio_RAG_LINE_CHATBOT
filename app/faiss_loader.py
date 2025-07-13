import faiss, pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# ============ EMBEDDER ============
embedder = SentenceTransformer("intfloat/multilingual-e5-base")


# ============ LOAD FAISS ============
def load_faiss_and_id2text(index_path, id2text_path, id2metadata_path):
    index = faiss.read_index(index_path)
    with open(id2text_path, "rb") as f:
        id2text = pickle.load(f)
    with open(id2metadata_path, "rb") as f:
        id2meta = pickle.load(f)
    return index, id2text, id2meta

index_hotlab, id2text_hotlab, id2meta_hotlab = load_faiss_and_id2text("database/index_hotlab.index", "database/id2text_hotlab.pkl", "database/id2meta_hotlab.pkl")
index_protocal, id2text_protocal, id2meta_protocal = load_faiss_and_id2text("database/index_protocal.index", "database/id2text_protocal.pkl", "database/id2meta_protocal.pkl")


# ============ SEARCH FAISS ============
def search_faiss(query, index, id2text, k=3):
    embedding = embedder.encode([query])
    D, I = index.search(np.array(embedding), k)
    top_chunks = [id2text[i] for i in I[0]]
    context = "\n\n".join(top_chunks)
    return context, top_chunks
