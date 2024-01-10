
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, JSONLoader, PDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from apps.model_factory import ModelFactory
import os
from apps.base import Task,function_stats
from typing import Any

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



class Retriever(Task):
    index_path = "./index.faiss"
    def __init__(self):
        self.embeddings = self.excurtor[0]

        if os.path.exists(self.file_path):
             self.vector_store = FAISS.load_local(self.index_path, self.excurtor[0])
        else:
            index = faiss.IndexFlatL2(1024)  # 使用L2距离度量
            self.vector_store = FAISS(index)

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
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return text_splitter.split_documents(documents)

    def build_vector_store(self, texts):
        self.vector_store.from_documents(texts, self.excurtor[0])

    def retrieve_documents(self, query, k=5):
        return self.vector_store.retrieve(query, k)
    
    @function_stats
    def run(self,input: Any=None,**kwargs):
        if input is None or input == "":
            return ""
        docs = self.load_documents(input)
        pages = self.split_documents(docs)
        self.build_vector_store(pages)
        return ""
    
    async def arun(self,input: Any=None,**kwargs):
        return self.run(input,**kwargs)
    
    def init_model(self):
        model = ModelFactory.get_model("embedding")
        return [model]

