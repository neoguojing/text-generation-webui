from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
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

class Qwen2Audio(BaseChatModel,CustomerLLM):
    model_path: str = Field(None, alias='model_path')
    max_window_size: Optional[int]   = 8192
    stop = ["Observation:", "Observation:\n","\nObservation:","Observation"]
    react_stop_words_tokens: Optional[List[List[int]]]
    stop_words_ids: Optional[List[List[int]]]
    online: bool = False
    token: str = ""
    stopping_criteria: StoppingCriteriaList = None

    def __init__(self, model_path: str,**kwargs):
        if model_path is not None:
            super(Qwen2Audio, self).__init__(llm=Qwen2AudioForConditionalGeneration.from_pretrained(
                model_path,
                device_map="auto",
            ))
            self.model_path = model_path
            self.processor = AutoProcessor.from_pretrained(model_path)
        else:
            # model_id = "Qwen/Qwen2-7B-Instruct"
            model_id = "Qwen/Qwen2-Audio-7B-Instruct"
            super(Qwen2Audio, self).__init__(llm=Qwen2AudioForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto",
            ))
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.processor.save_pretrained(os.path.join(model_root,"qwen2-audio"))
            self.model.save_pretrained(os.path.join(model_root,"qwen2-audio"))
            
       
        self.model.to(self.device)

    @property
    def _llm_type(self) -> str:
        return "qwen2-audio"
    
    @property
    def model_name(self) -> str:
        return "qwen2-audio"
    

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:

        conversation = self._format_message(messages)
        print("qwen2 audio input-----------:",input)
        
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios = []
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audios.append(librosa.load(
                            BytesIO(urlopen(ele['audio_url']).read()), 
                            sr=self.processor.feature_extractor.sampling_rate)[0]
                        )

        inputs = self.processor(text=text, audios=audios, return_tensors="pt", padding=True)
        inputs.input_ids = inputs.input_ids.to(self.device)

        generate_ids = self.model.generate(**inputs, max_length=256)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
        print("qwen2audio output-----------:",response)
        message = AIMessage(
            content=response,
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
        print("qwen2 messages:",messages)

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs =self.tokenizer([text], return_tensors="pt").to(self.device)

        print("qwen2 input_ids:",model_inputs.input_ids)
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=self.max_window_size
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)
        return response
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_path": self.model_path}

    def _format_message(self,messages: List[BaseMessage]) -> List[Dict]:
        ret = []
        for item in messages:
            if isinstance(item,SystemMessage):
                ret.append({"role": "system", "content": item.content})
            if isinstance(item,AIMessage):
                ret.append({"role": "assistant", "content": item.content})
            if isinstance(item,HumanMessage):
                user_content = []
                if looks_like_path(item.content):
                    user_content.append({"type": "audio", "audio_url": item.content})
                else:
                    user_content.append({"type": "text", "text": item.content})
                ret.append({"role": "user", "content": user_content})
        return ret
    
from pathlib import Path

def looks_like_path(path_str):
    try:
        p = Path(path_str)
        return p.parts != (path_str,)
    except ValueError:
        return False


if __name__ == '__main__':
    prompt = '''
    俄乌战争
    '''
    model = Qwen2Audio(model_path=None)
    # out = model._call(prompt,system="你是一个政治专家,请使用中文",history=[["二战","不知道"]])
    input = HumanMessage(content=prompt)
    out = model._generate([input],system="你是一个政治专家,请使用中文",history=[["二战","不知道"]])
    print(out)
