import sys
import os
# 获取当前脚本所在的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将当前package的父目录作为顶层package的路径
top_package_path = os.path.abspath(os.path.join(current_dir, ".."))

# 将顶层package路径添加到sys.path
sys.path.insert(0, top_package_path)
from langchain_community.llms import OpenAI,Tongyi
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
import threading
import gc
from langchain_community.chat_models import QianfanChatEndpoint
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# openai
os.environ['OPENAI_API_KEY'] = ''
# qianfan
os.environ["QIANFAN_AK"] = "your_ak"
os.environ["QIANFAN_SK"] = "your_sk"
# tongyi
os.environ["DASHSCOPE_API_KEY"] = ""

class ModelFactory:
    _instances = {}
    _lock = threading.Lock()

    @staticmethod
    def get_model(model_name, model_path=""):
        """获取模型实例，并缓存"""
        if model_name not in ModelFactory._instances or ModelFactory._instances[model_name] is None:
            with ModelFactory._lock:
                if model_name not in ModelFactory._instances or ModelFactory._instances[model_name] is None:
                    ModelFactory._instances[model_name] = ModelFactory._load_model(model_name, model_path)
        return ModelFactory._instances[model_name]

    @staticmethod
    def _load_model(model_name, model_path=""):
        """实际负责模型加载的私有方法"""
        print(f"Loading the model {model_name}, wait a minute...")
        if model_name == "openai":
            return OpenAI()
        elif model_name == "qwen2": 
            model_path = os.path.join(model_root,"qwen2")
            return Qwen2Chat(model_path=model_path)
        elif model_name == "qianfan": 
            return QianfanChatEndpoint(streaming=True, model="ERNIE-Bot-4")
        elif model_name == "tongyi": 
            return Tongyi()
        elif model_name == "llama3": 
            model_path = os.path.join(model_root,"llama3")
            return Llama3Chat(model_path=model_path,token=None)
        elif model_name == "translate": 
            model_path = os.path.join(model_root,"nllb/")
            return Translate(model_path=model_path)
        elif model_name == "speech": 
            return SeamlessM4t()
        elif model_name == "text2image": 
            return StableDiff()
        elif model_name == "image2image": 
            return Image2Image()
        elif model_name == "speech2text": 
            return Whisper()
        elif model_name == "text2speech": 
            return  XTTS()
        elif model_name == "embedding": 
            return Embedding()
        else:
            raise ValueError(f"Invalid model name: {model_name}")

    @staticmethod
    def destroy(model_name):
        """销毁模型实例"""
        if model_name in ModelFactory._instances and ModelFactory._instances[model_name] is not None:
            with ModelFactory._lock:
                instance = ModelFactory._instances.pop(model_name, None)
                if instance:
                    ModelFactory._safe_destroy(instance)

    @staticmethod
    def _safe_destroy(instance):
        """安全销毁模型实例"""
        refcount = len(gc.get_referrers(instance))
        if refcount <= 2:
            if isinstance(instance, CustomerLLM):
                instance.destroy()
            del instance
            gc.collect()

    @staticmethod
    def release():
        """释放所有模型实例"""
        for model_name in list(ModelFactory._instances.keys()):
            ModelFactory.destroy(model_name)