import sys
import os
# 获取当前脚本所在的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将当前package的父目录作为顶层package的路径
top_package_path = os.path.abspath(os.path.join(current_dir, ".."))

# 将顶层package路径添加到sys.path
sys.path.insert(0, top_package_path)
from langchain_community.llms import OpenAI,Tongyi
from langchain_community.llms import HuggingFacePipeline
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain_core.language_models import BaseChatModel
from apps.translate.nllb import Translate
from apps.multi_task.speech import SeamlessM4t,Whisper,XTTS
# from apps.multi_task.speech import SeamlessM4t,Whisper
from apps.image.sd import StableDiff,Image2Image
from apps.llama.llama3 import Llama3
from apps.qwen2.qwen2_chat import Qwen2Chat
from apps.llama.llama3_chat import Llama3Chat
from apps.embedding.embedding import Embedding
from apps.config import model_root
from apps.base import CustomerLLM
from pydantic import  Field, root_validator
import torch
import threading
import gc
import weakref
import pdb

from langchain_core.messages import BaseMessage, HumanMessage,AIMessage,SystemMessage
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult,LLMResult
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# openai
os.environ['OPENAI_API_KEY'] = ''
# qianfan
os.environ["QIANFAN_AK"] = "your_ak"
os.environ["QIANFAN_SK"] = "your_sk"
# tongyi
os.environ["DASHSCOPE_API_KEY"] = ""

class ModelFactory:
    temperature = 0
    _instances = {}
    _lock = threading.Lock()  # 异步锁

    @staticmethod
    def get_model(model_name,model_path=""):
        if model_name not in ModelFactory._instances or ModelFactory._instances[model_name] is None:
            with ModelFactory._lock:
                if model_name not in ModelFactory._instances or ModelFactory._instances[model_name] is None:
                    print(f"loading the model {model_name},wait a minute...")
                    if model_name == "openai":
                        instance = OpenAI()
                    elif model_name == "qwen2": 
                        model_path = os.path.join(model_root,"qwen2")
                        instance = Qwen2Chat(model_path=model_path)
                    elif model_name == "qianfan": 
                        instance = QianfanChatEndpoint(streaming=True, model="ERNIE-Bot-4")
                    elif model_name == "tongyi": 
                        instance = Tongyi()
                    elif model_name == "llama3": 
                        model_path = os.path.join(model_root,"llama3")
                        # instance = Llama3(model_path=model_path)
                        instance = Llama3Chat(model_path=model_path,token=None)
                    elif model_name == "translate": 
                        model_path = os.path.join(model_root,"nllb/")
                        instance = Translate(model_path=model_path)
                    elif model_name == "speech": 
                        instance = SeamlessM4t()
                    elif model_name == "text2image": 
                        instance = StableDiff()
                    elif model_name == "image2image": 
                        instance = Image2Image()
                    elif model_name == "speech2text": 
                        instance = Whisper()
                    elif model_name == "text2speech": 
                        instance = XTTS()
                    elif model_name == "embedding": 
                        instance = Embedding()
                    else:
                        raise Exception("Invalid model name")
                    
                    ModelFactory._instances[model_name] = instance
                    print(f"model {model_name} load finished...")
                    
        return  ModelFactory._instances[model_name]
    
    @staticmethod
    def destroy(model_name):
        if model_name in ModelFactory._instances and ModelFactory._instances[model_name] is not None:
            with ModelFactory._lock:
                if model_name in ModelFactory._instances and ModelFactory._instances[model_name] is not None:
                    obj = ModelFactory._instances.get(model_name)
                    refcount = len(gc.get_referrers(obj))
                    # print(f"{type(obj)} refer by {refcount} object")
                    if refcount <= 2:
                        ModelFactory._instances[model_name] = None
                        refcount = len(gc.get_referrers(obj))
                        # print(f"----{type(obj)} refer by {refcount} object")
                        if isinstance(obj, CustomerLLM) and refcount <= 1 :
                            obj.destroy()

    @staticmethod
    def release():
        for k, model in ModelFactory._instances.items():
            ModelFactory.destroy(k)