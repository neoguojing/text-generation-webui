from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, JSONLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
from model_factory import ModelFactory
import os
from base import Task,function_stats
from typing import Any

class Retriever(Task):
    index_path = "./index.faiss"
    def __init__(self):
        self.embeddings = self.excurtor[0]

        if os.path.exists(self.index_path):
             self.vector_store = FAISS.load_local(self.index_path, self.excurtor[0])
             print("load faiss from local index ")
        else:
            index = faiss.IndexFlatL2(1024)
            self.vector_store = FAISS(self.embeddings,index,InMemoryDocstore(),{})
            self.vector_store.save_local("./","index.faiss")
        
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"score_threshold": 0.5,"k": 1}
            )

    def load_documents(self, file_paths):
        documents = []

        if not isinstance(file_paths, list):
            file_paths = [file_paths]

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
        self.vector_store.add_documents(texts)

    def retrieve_documents(self, query):
        return self.retriever.get_relevant_documents(query)
    
    @function_stats
    def run(self, input: Any,**kwargs):
        if input is None or input == "":
            return 
        docs = self.load_documents(input)
        print("```````````",docs)
        pages = self.split_documents(docs)
        print("```````````",pages)
        self.build_vector_store(pages)

        self.vector_store.save_local("./","index.faiss")

    
    async def arun(self,input: Any=None,**kwargs):
        return self.run(input,**kwargs)
    
    def init_model(self):
        model = ModelFactory.get_model("embedding")
        return [model]

if __name__ == '__main__':
    r = Retriever()
    r.run("../requirements_amd.txt")
    docs = r.retrieve_documents('python_version == "3.11"')
    print(docs)
