from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, JSONLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
from .model_factory import ModelFactory
import os
from .base import Task,function_stats
from typing import Any

class Retriever(Task):
    index_path = "./"
    index_name = "default"
    batch_size = 16
    def __init__(self):
        self.embeddings = self.excurtor[0]
        if os.path.exists(self.index_path+self.index_name+".faiss"):
             print("load faiss from local index ")
             self.vector_store = FAISS.load_local(self.index_path, self.excurtor[0],self.index_name,allow_dangerous_deserialization=True)
             
        else:
            index = faiss.IndexFlatL2(1024)
            self.vector_store = FAISS(self.embeddings,index,InMemoryDocstore(),{})
            self.vector_store.save_local(self.index_path,self.index_name)
        
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

    def build_vector_store(self, docs):
        self.vector_store.add_documents(docs)

    def retrieve_documents(self, query):
        docs = self.retriever.get_relevant_documents(query)
        texts = []
        for d in docs:
            texts.append(d.page_content)
        return texts
    
    @function_stats
    def run(self, input: Any,**kwargs):
        if input is None or input == "":
            return 
        
        docs = self.load_documents(input)
        print("load_documents:",len(docs))
        pages = self.split_documents(docs)
        print("split_documents:",len(pages))
        groups = []
        if len(pages) > self.batch_size:
            groups = [docs[i:i+self.batch_size] for i in range(0, len(docs), self.batch_size)]
        else:
            groups = [pages]
        print("groups:",len(groups))
        for g in groups:
            self.build_vector_store(g)

        self.vector_store.save_local(self.index_path,self.index_name)

    
    async def arun(self,input: Any=None,**kwargs):
        return self.run(input,**kwargs)
    
    def init_model(self):
        model = ModelFactory.get_model("embedding")
        return [model]

# if __name__ == '__main__':
#     r = Retriever()
#     # r.run("../requirements_amd.txt")
#     docs = r.retrieve_documents('python_version == "3.11"')
#     print(docs)
