from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import LogitsProcessorList
from transformers import GenerationConfig,StoppingCriteriaList
import torch
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage, HumanMessage,AIMessage,SystemMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult,LLMResult
from langchain_core.language_models import BaseChatModel
import os
import sys
# 获取当前脚本所在的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将当前package的父目录作为顶层package的路径
top_package_path = os.path.abspath(os.path.join(current_dir, "../../"))

# 将顶层package路径添加到sys.path
sys.path.insert(0, top_package_path)
from apps.generation_utils import StopWordsLogitsProcessor,StopStringCriteria
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
import pdb

class Llama3Chat(BaseChatModel,CustomerLLM):
    model_path: str = Field(None, alias='model_path')
    max_window_size: Optional[int]   = 8192
    stop = ["Observation:", "Observation:\n","\nObservation:","Observation"]
    react_stop_words_tokens: Optional[List[List[int]]]
    stop_words_ids: Optional[List[List[int]]]
    online: bool = False
    token: str = ""
    stopping_criteria: StoppingCriteriaList = None

    def __init__(self, model_path: str,token: str,**kwargs):
        if model_path is not None:
            super(Llama3Chat, self).__init__(llm=AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            ))
            self.model_path = model_path
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
            super(Llama3Chat, self).__init__(llm=AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            ))
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            # tokenizer.save_pretrained(os.path.join(model_root,"llama3"))
            # model.save_pretrained(os.path.join(model_root,"llama3"))
            
       
        self.model.to(self.device)
        self.react_stop_words_tokens = []
        self.stop_words_ids = [self.tokenizer.encode(stop_) for stop_ in self.stop]
        self.react_stop_words_tokens.append(self.tokenizer.eos_token_id)
        self.react_stop_words_tokens.append(self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        print("<|eot_id|>",self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        print("stop_words_ids:",self.stop_words_ids)
        # pdb.set_trace()
        self.stopping_criteria = StoppingCriteriaList([StopStringCriteria(tokenizer=self.tokenizer, stop_strings=self.stop)])
        

    @property
    def _llm_type(self) -> str:
        return "llama3"
    
    @property
    def model_name(self) -> str:
        return "llama3"
    

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # pdb.set_trace()
        logits_processor = None
        if stop is not None:
            self.stop_words_ids.extend([self.tokenizer.encode(stop_) for stop_ in stop])
            
        if self.stop_words_ids is not None:
            stop_words_logits_processor = StopWordsLogitsProcessor(
                stop_words_ids=self.stop_words_ids,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            logits_processor = LogitsProcessorList([stop_words_logits_processor])

        

        generation_config = GenerationConfig(
            max_new_tokens=self.max_window_size,
            eos_token_id=self.react_stop_words_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            num_beams=1,
            temperature=0.6,
            top_p=0.9,
        )
        
        input = self._format_message(messages)
        print("Llama3 input-----------:",input)
        
        input_ids =self.tokenize(input)
        # print("Llama3 input:",input)
        outputs = self.model.generate(
            input_ids,
            generation_config = generation_config,
            # logits_processor=logits_processor,
            stopping_criteria=self.stopping_criteria,
        )

        response = outputs[0][input_ids.shape[-1]:]
        decode_resp = self.tokenizer.decode(response, skip_special_tokens=True)
    
        print("Llama3 output-----------:",decode_resp)
        message = AIMessage(
            content=decode_resp,
            additional_kwargs={},  # Used to add additional payload (e.g., function calling request)
            response_metadata={  # Use for response metadata
                "time_in_seconds": 3,
            },
        )

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

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

        input_ids =self.tokenize(messages)
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

    def tokenize(self,messages: Any):
        input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)
        return input_ids

    def _format_message(self,
                        messages: List[BaseMessage]) -> List[Dict]:
        ret = []
        for item in messages:
            if isinstance(item,SystemMessage):
                ret.append({"role": "system", "content": item.content})
            if isinstance(item,AIMessage):
                ret.append({"role": "assistant", "content": item.content})
            if isinstance(item,HumanMessage):
                ret.append({"role": "user", "content": item.content})

        return ret
    
    def _api_call(self,input: List[BaseMessage]):
        import requests
        API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
        headers = {"Authorization": f"Bearer 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"}
        # headers = {"Authorization": f"Bearer {self.token}"}

        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()
            
        output = query({
            "inputs": input,
        })

        return output[0]['generated_text']