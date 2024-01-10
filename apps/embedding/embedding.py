from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import normalize
import torch
import os
from typing import Any, List, Mapping, Optional,Union
from langchain.callbacks.manager import (
    CallbackManagerForLLMRun
)
from pydantic import  Field
from apps.base import Task,CustomerLLM
from apps.config import model_root

class Embedding(CustomerLLM):
    model_path: str = Field(None, alias='model_path')

    def __init__(self, model_path: str=os.path.join(model_root,"acge-large-zh"),**kwargs):
        super(Embedding, self).__init__(
            llm=AutoModel.from_pretrained(
                os.path.join(model_root,"acge-large-zh")
        ))
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained('acge-large-zh')
        self.tokenizer.save_pretrained(os.path.join(model_root,"nllb"))
        self.model.save_pretrained(os.path.join(model_root,"nllb"))

    @property
    def _llm_type(self) -> str:
        return "acge-large-zh"
    
    @property
    def model_name(self) -> str:
        return "embedding"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str | any:
        batch_data = self.tokenizer(
                batch_text_or_text_pairs=prompt,
                padding="longest",
                return_tensors="pt",
                max_length=1024,
                truncation=True,
        )

        attention_mask = batch_data["attention_mask"]
        model_output = self.model(**batch_data)
        last_hidden = model_output.last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        vectors = normalize(vectors, norm="l2", axis=1, )
        print(vectors.shape) 
        return vectors

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_path": self.model_path}
    
