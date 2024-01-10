from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS

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


