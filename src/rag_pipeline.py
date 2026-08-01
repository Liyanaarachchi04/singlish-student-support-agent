import os
from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

CHROMA_PATH = "data/chroma_db"
DATA_PATH = "data/corpus"

def get_embedding_function():
    """Returns standard huggingface embeddings for semantic search."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_and_split_documents() -> List[Document]:
    """Loads documents from corpus directory and splits them into chunks."""
    loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def initialize_vector_store() -> Chroma:
    """Initializes or loads the Chroma vector database."""
    embeddings = get_embedding_function()
    
    # Check if vector DB already exists
    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    else:
        chunks = load_and_split_documents()
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PATH
        )
    return vector_store

def query_rag_system(query: str, k: int = 3) -> List[Document]:
    """Retrieves relevant chunks from Chroma for a student query."""
    vector_store = initialize_vector_store()
    results = vector_store.similarity_search(query, k=k)
    return results

def evaluate_retrieval_queries():
    """Runs 5 sample Singlish queries to test RAG retrieval accuracy."""
    test_queries = [
        "Repeat exam register karanne kohomada?",
        "IT41043 prerequisite subjects monawada?",
        "Medical submit karanna form eka thiyenne koheda?",
        "Late registration fee kiyada?",
        "Hostel and scholarship issue kata kiyannada?"
    ]
    
    print("=== RUNNING RAG RETRIEVAL EVALUATION ===")
    for i, q in enumerate(test_queries, 1):
        print(f"\n[Query {i}]: {q}")
        results = query_rag_system(q, k=2)
        for idx, doc in enumerate(results, 1):
            snippet = doc.page_content.replace("\n", " ")[:150]
            print(f"  Snippet {idx}: {snippet}...")

if __name__ == "__main__":
    evaluate_retrieval_queries()