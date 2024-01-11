from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, JSONLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from model_factory import ModelFactory
import os
from base import Task,function_stats
from typing import Any

class Retriever(Task):
    index_path = "./index.faiss"
    def __init__(self):
        self.embeddings = self.excurtor[0]

        if os.path.exists(self.file_path):
             self.vector_store = FAISS.load_local(self.index_path, self.excurtor[0])
        else:
            index = faiss.IndexFlatL2(1024)
            self.vector_store = FAISS(index)

    def load_documents(self, file_paths):
        documents = []
        for file_path in file_paths:
            if file_path.endswith('.txt'):
                self.loader = TextLoader(file_path)
            elif file_path.endswith('.json'):
                self.loader = JSONLoader(file_path)
            elif file_path.endswith('.pdf'):
                self.loader = PyPDFLoader(file_path)
            else:
                raise ValueError("Unsupported file format")
            documents.extend(self.loader.load())
        return documents

    def split_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return text_splitter.split_documents(documents)

    def build_vector_store(self, texts):
        self.vector_store.from_documents(texts, self.excurtor[0])

    def retrieve_documents(self, query, k=5):
        return self.vector_store.retrieve(query, k)
    
    @function_stats
    def run(self, input: Any,**kwargs):
        if input is None or input == "":
            return 
        docs = self.load_documents(input)
        print("```````````",docs)
        pages = self.split_documents(docs)
        print("```````````",pages)
        self.build_vector_store(pages)
    
    async def arun(self,input: Any=None,**kwargs):
        return self.run(input,**kwargs)
    
    def init_model(self):
        model = ModelFactory.get_model("embedding")
        return [model]

if __name__ == '__main__':
    r = Retriever()
    r.run("../requirements_amd.txt")