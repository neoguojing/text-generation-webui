from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import normalize
import torch
import os
import sys
# 获取当前脚本所在的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将当前package的父目录作为顶层package的路径
top_package_path = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.insert(0, top_package_path)
from typing import Any, List, Mapping, Optional,Union
from langchain.callbacks.manager import (
    CallbackManagerForLLMRun
)
from pydantic import  Field
from apps.base import Task,CustomerLLM
from apps.config import model_root
from langchain_core.embeddings import Embeddings

class Embedding(Embeddings,CustomerLLM):
    model_path: str = Field(None, alias='model_path')

    def __init__(self, model_path: str=os.path.join(model_root,"acge-large-zh"),**kwargs):
        if model_path is not None:
            super(Embedding, self).__init__(
                llm=AutoModel.from_pretrained(
                    os.path.join(model_root,"acge-large-zh")
            ))
            # super(Embedding, self).__init__(
            #     llm=AutoModel.from_pretrained('aspire/acge-large-zh'))
            # self.tokenizer = AutoTokenizer.from_pretrained('aspire/acge-large-zh')
            # self.tokenizer.save_pretrained(os.path.join(model_root,"acge-large-zh"))
            # self.model.save_pretrained(os.path.join(model_root,"acge-large-zh"))
            self.model_path = model_path
            self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_root,"acge-large-zh"))
        else:
            super(Embedding, self).__init__(
                 llm=AutoModel.from_pretrained('aspire/acge-large-zh'))
            self.tokenizer = AutoTokenizer.from_pretrained('aspire/acge-large-zh')
        # self.model.to(self.device)
       
    @property
    def _llm_type(self) -> str:
        return "aspire/acge-large-zh"
    
    @property
    def model_name(self) -> str:
        return "embedding"
    
    def _call(
        self,
        prompt: Union[str,List[str]],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        batch_data = self.tokenizer(
                text=prompt,
                padding="longest",
                return_tensors="pt",
                max_length=1024,
                truncation=True,
        )

        attention_mask = batch_data["attention_mask"]
        # batch_data.to('cuda')
        model_output = self.model(**batch_data)
        # model_output = model_output.cpu()
        last_hidden = model_output.last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        
        vectors = vectors.detach().numpy()
        # 对每行的向量进行归一化
        vectors = normalize(vectors, norm="l2", axis=1)
        print("_call",vectors.shape) 
        return vectors

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_path": self.model_path}
    
    def embed_documents(self, texts) -> List[List[float]]:
        # Embed a list of documents
        embeddings = []
        print("embed_documents:",len(texts),type(texts))
        embedding = self._call(texts)
        for row in embedding:
            embeddings.append(row)
        return embeddings
    
    def embed_query(self, text) -> List[float]:
        # Embed a single query
        embedding = self._call(text)
        return embedding[0]
    

# if __name__ == '__main__':
#     sd = Embedding()
#     v1 = sd.embed_query("That is a happy person")
#     v2 = sd.embed_query("That is a very happy person")
#     print(v1 @ v2.T)
