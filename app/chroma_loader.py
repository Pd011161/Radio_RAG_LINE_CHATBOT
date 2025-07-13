from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_function = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

hotlab_vector_store = Chroma(
    persist_directory="chroma_langchain",
    embedding_function=embedding_function,
    collection_name="hotlab_collection"
)

protocol_vector_store = Chroma(
    persist_directory="chroma_langchain",
    embedding_function=embedding_function,
    collection_name="protocol_collection"
)

#  todo
# bmd_vector_store = Chroma(
#     persist_directory="/Users/Aksorn_AI/Desktop/RAG/chroma_langchain",
#     embedding_function=embedding_function,
#     collection_name=""
# )

iodine_vector_store = Chroma(
    persist_directory="chroma_langchain",
    embedding_function=embedding_function,
    collection_name="iodine_collection"
)


def search_chroma(query, vector_store):
    # Retriever
    retriever = vector_store.as_retriever(
    search_type="similarity",
    # search_type="mmr",
    search_kwargs={
            "k": 5,
            # "fetch_k": 20,
            # "filter": {
            #     "sub-topic": "" 
            # }
        }
    )
    # Retrieval
    docs = retriever.get_relevant_documents(query)
    # Context
    context = "\n\n".join(doc.page_content for doc in docs)
    return context



