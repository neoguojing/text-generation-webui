from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import os
import sys
# 获取当前脚本所在的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将当前package的父目录作为顶层package的路径
top_package_path = os.path.abspath(os.path.join(current_dir, "../../"))

# 将顶层package路径添加到sys.path
sys.path.insert(0, top_package_path)
from apps.config import model_root
from apps.base import Task,CustomerLLM
from typing import (
    AbstractSet,
    cast,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Sequence,
    TypedDict,
    Union,
)

Role = Literal["system", "user", "assistant"]
class Message(TypedDict):
    role: Role
    content: str

class Llama3(CustomerLLM):
    model_path: str = Field(None, alias='model_path')
    chat_format: Optional[str]   = 'chatml'
    max_window_size: Optional[int]   = 8192
    stop = ["Observation:", "Observation:\n","\nObservation:"]
    react_stop_words_tokens: Optional[List[List[int]]]
    terminators:  Optional[List]

    def __init__(self, model_path: str,**kwargs):

        super(Llama3, self).__init__(llm=AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ))
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

        # tokenizer = AutoTokenizer.from_pretrained(model_id)
        # tokenizer.save_pretrained(os.path.join(model_root,"llama3"))
        # model = AutoModelForCausalLM.from_pretrained(
        #     model_id,
        #     torch_dtype=torch.bfloat16,
        #     device_map="auto",
        # )

        # model.save_pretrained(os.path.join(model_root,"llama3"))
        
       
        self.model.to(self.device)
        self.react_stop_words_tokens = [self.tokenizer.encode(stop_) for stop_ in self.stop]
        self.react_stop_words_tokens.append(self.tokenizer.eos_token_id)
        self.react_stop_words_tokens.extend(self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        

    @property
    def _llm_type(self) -> str:
        return "llama3"
    
    @property
    def model_name(self) -> str:
        return "llama3"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            self.react_stop_words_tokens.extend([self.tokenizer.encode(stop_) for stop_ in stop])
        
        system = kwargs.pop('system', '')
        history = kwargs.pop('history', [])
        print("Llama3 history:",history)
        print("Llama3 system:",system)
        print("Llama3 prompt:",prompt)

        messages = []
        messages.append(Message = {"role": "system", "content": system})
        for row in history:
            if row is None or len(row) < 2:
                continue
            messages.append(Message = {"role": "user", "content": row[0]})
            messages.append(Message = {"role": "assistant", "content": row[1]})
        messages.append(Message = {"role": "user", "content": prompt})

        input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=react_stop_words_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_path": self.model_path}
