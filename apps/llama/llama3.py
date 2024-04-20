from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from langchain.callbacks.manager import CallbackManagerForLLMRun
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
    Optional,
    Any,
    Mapping,
)
from pydantic import  Field

Role = Literal["system", "user", "assistant"]
class Message(TypedDict):
    role: Role
    content: str

class Llama3(CustomerLLM):
    model_path: str = Field(None, alias='model_path')
    max_window_size: Optional[int]   = 8192
    stop = ["Observation:", "Observation:\n","\nObservation:"]
    react_stop_words_tokens: Optional[List[List[int]]]
    stop_words_ids: Optional[List[List[int]]]

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
        self.react_stop_words_tokens = []
        self.stop_words_ids = [self.tokenizer.encode(stop_) for stop_ in self.stop]
        self.react_stop_words_tokens.append(self.tokenizer.eos_token_id)
        self.react_stop_words_tokens.append(self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        print("<|eot_id|>",self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        

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
            self.stop_words_ids.extend([self.tokenizer.encode(stop_) for stop_ in stop])
        
        system = kwargs.pop('system', '')
        history = kwargs.pop('history', [])
        
        messages = []
        messages.append({"role": "system", "content": system})
        for row in history:
            if row is None or len(row) < 2:
                continue
            messages.append({"role": "user", "content": row[0]})
            messages.append({"role": "assistant", "content": row[1]})
        messages.append({"role": "user", "content": prompt})
        print("Llama3 messages:",messages)

        input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)
        print("Llama3 input_ids:",self.react_stop_words_tokens)
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_window_size,
            eos_token_id=self.react_stop_words_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            # stop_words_ids=self.stop_words_ids,
        )
        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_path": self.model_path}


# if __name__ == '__main__':
#     prompt = '''
#     俄乌战争
#     '''
#     model = Llama3(model_path=os.path.join(model_root,"llama3"))
#     out = model._call(prompt,system="你是一个政治专家,请使用中文",history=[["二战","不知道"]])
#     print(out)
