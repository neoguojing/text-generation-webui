
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, JSONLoader, PDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def loader(path: str):
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    return db

def search(query: str):
    retriever = db.as_retriever(
        search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}
    )
    docs = retriever.get_relevant_documents(query)
    return docs



class Retriever:
    def __init__(self, vector_store):
        index = faiss.IndexFlatL2(d)  # 使用L2距离度量
        vector_store = FAISS(index)
        self.vector_store = vector_store

    def load_documents(self, file_paths):
        documents = []
        for file_path in file_paths:
            if file_path.endswith('.txt'):
                loader = TextLoader(file_path)
            elif file_path.endswith('.json'):
                loader = JSONLoader(file_path)
            elif file_path.endswith('.pdf'):
                loader = PDFLoader(file_path)
            else:
                raise ValueError("Unsupported file format")
            documents.extend(loader.load())
        return documents

    def split_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        return text_splitter.split_documents(documents)

    def build_vector_store(self, texts):
        self.vector_store.from_documents(texts)

    def retrieve_documents(self, query, k=5):
        return self.vector_store.retrieve(query, k)

