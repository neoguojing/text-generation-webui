from langchain.tools import tool
from langchain import hub
from langchain.chains.llm import LLMChain
from langchain.agents import AgentExecutor, LLMSingleActionAgent
from langchain_community.utilities import SerpAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain_community.utilities import ArxivAPIWrapper
from langchain.utilities.alpha_vantage import AlphaVantageAPIWrapper
from langchain.agents import Tool,create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
import os
import sys
import time
from typing import Any
from .tools import get_stock
# 获取当前脚本所在的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将当前package的父目录作为顶层package的路径
top_package_path = os.path.abspath(os.path.join(current_dir, ".."))

# 将顶层package路径添加到sys.path
sys.path.insert(0, top_package_path)
import threading
from apps.base import Task,function_stats
from apps.model_factory import ModelFactory
from apps.prompt import QwenAgentPromptTemplate,translate_prompt,AgentPromptTemplate
from apps.parser import QwenAgentOutputParser
from .retriever import Retriever

TASK_AGENT = 100
TASK_TRANSLATE = 200
TASK_DATA_HANDLER = 300
TASK_IMAGE_GEN = 400
TASK_SPEECH = 500
TASK_GENERAL = 600
TASK_RETRIEVER = 700


os.environ['SERPAPI_API_KEY'] = 'f765e0536e1a72c2f353bb1946875937b3ac7bed0270881f966d4147ac0a7943'
os.environ['WOLFRAM_ALPHA_APPID'] = 'QTJAQT-UPJ2R3KP89'
os.environ["ALPHAVANTAGE_API_KEY"] = '1JXIUYN26HYID5Y9'

search = SerpAPIWrapper()
# search = DuckDuckGoSearchRun()
WolframAlpha = WolframAlphaAPIWrapper()
arxiv = ArxivAPIWrapper()
# alpha_vantage = AlphaVantageAPIWrapper()


@tool("image generate", return_direct=True)
def image_gen(input:str) ->str:
    """Useful for when you need to generate or draw a picture by input text.
    Text to image diffusion model capable of generating photo-realistic images given any text input."""
    task = TaskFactory.create_task(TASK_IMAGE_GEN)
    return task.run(input)

@tool("speech or audio generate", return_direct=True)
def text2speech(input:str) ->str:
    """Useful for when you need to transfer text to speech or audio.Speech to speech translation.Speech to text translation.Text to speech translation.Text to text translation.Automatic speech recognition."""
    task = TaskFactory.create_task(TASK_SPEECH)
    return task.run(input)



tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for when you need to answer questions about current events"
    ),
    # Tool(
    #     name="Math",
    #     func=WolframAlpha.run,
    #     description="Useful for when you need to answer questions about Math, Science, Technology, Culture, Society and Everyday Life."
    # ),
    # Tool(
    #     name="arxiv",
    #     func=arxiv.run,
    #     description="A wrapper around Arxiv.org Useful for when you need to answer questions about Physics, Mathematics, Computer Science, \
    #         Quantitative Biology, Quantitative Finance, Statistics, Electrical Engineering, and Economics from scientific articles \
    #         on arxiv.org."
    # ),
    # Tool(
    #     name="alphaVantage",
    #     func=alpha_vantage.run,
    #     description ="Alpha Vantage is a platform useful for provides financial market data and related services. It offers a wide range \
    #           of financial data, including stock market data, cryptocurrency data, and forex data. Developers can access real-time and \
    #             historical market data through Alpha Vantage, enabling them to perform technical analysis, modeling, and develop financial\
    #             applications."
    # ),
    image_gen,
    text2speech,
    get_stock,
]


class CustomAgent(LLMSingleActionAgent):
    def plan(self, intermediate_steps, **kwargs):
        # Implement your custom logic here
        # based on intermediate_steps and kwargs

        # # Example logic:
        # if intermediate_steps:
        #     # Get the last intermediate step
        #     last_step = intermediate_steps[-1]

        #     # Check if the last step is a specific tool
        #     if last_step[0].tool == "TranslateTool":
        #         # Perform some action based on the translation
        #         translated_text = last_step[1]
        #         return AgentAction(tool="DrawPictureTool", tool_input=translated_text, log="")

        # # If no specific condition is met, return AgentFinish
        # return AgentFinish()
        # print("****************\n",intermediate_steps,kwargs.keys())
        return super().plan(intermediate_steps,**kwargs)

class Agent(Task):
    
    def __init__(self):
        # prompt = QwenAgentPromptTemplate(
        #     tools=tools,
        #     # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        #     # This includes the `intermediate_steps` variable because that is needed
        #     input_variables=["input", "intermediate_steps",'tools', 'tool_names', 'agent_scratchpad']
        # )
        # prompt = hub.pull("hwchase17/react-chat")

        prompt = AgentPromptTemplate(
            tools=tools,
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables=["input", "intermediate_steps",'tools', 'tool_names', 'agent_scratchpad']
        )

        print("agent prompt:",prompt)
        # from langchain.memory import ConversationBufferMemory
        # self.memory = ConversationBufferMemory(memory_key="history")
        
        output_parser = QwenAgentOutputParser()
        llm_chain = LLMChain(llm=self.excurtor[0], prompt=prompt)

        tool_names = [tool.name for tool in tools]

        # agent = CustomAgent(
        #     llm_chain=llm_chain,
        #     output_parser=output_parser,
        #     stop=["\nObservation:"],
        #     allowed_tools=tool_names,
        #     max_iterations=5,
        # )

        # self._executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

        agent = create_react_agent(
            llm=self.excurtor[0],
            tools=tools,
            prompt=prompt,
            output_parser=output_parser,
        )

        self._executor = AgentExecutor.from_agent_and_tools(agent=agent,tools=tools, verbose=True,
                                                            handle_parsing_errors=True,stream_runnable=False)

    @function_stats
    def run(self,input: Any=None,**kwargs):
        if input is None or input == "":
            return ""
        
        # print("Agent.run input---------------",input)
        output = self._executor.invoke({"input":input,"chat_history":""},**kwargs)
        # print("Agent.run output----------------------:",output)
        return output["output"]
    
    async def arun(self,input: Any=None,**kwargs):
        return self.run(input,**kwargs)
    
    def init_model(self):
        model = ModelFactory.get_model("qwen")
        # model = ModelFactory.get_model("llama3")
        return [model]
    
    def destroy(self):
        print("Agent model should not be destroy ")

class General(Task):
    def init_model(self):
        model = ModelFactory.get_model("qwen")
        return [model]   

import pdb
class ImageGenTask(Task):

    def init_model(self):
        model = ModelFactory.get_model("text2image")
        model1 = ModelFactory.get_model("image2image")
        return [model,model1]
    
    @function_stats
    def run(self,input:Any,**kwargs):
        if input is None:
            return ""
        image_path = kwargs.pop("image_path","")
        image_obj = kwargs.pop("image_obj",None)

        # translate = ModelFactory.get_model("qwen")
        translate = ModelFactory.get_model("llama3")
        en_prompt = translate_prompt(input)
        input = translate.predict(en_prompt)
        if image_path != "" or image_obj is not None:
            output = self.excurtor[1]._call(input,image_path=image_path,image_obj=image_obj)
        else:
            output = self.excurtor[0]._call(input,**kwargs)
        
        return output

class Speech(Task):
    def init_model(self):
        # model = ModelFactory.get_model("speech")
        # return [model]
        model = ModelFactory.get_model("speech2text")
        model1 = ModelFactory.get_model("text2speech")
        return [model,model1]
        
    
    @function_stats
    def run(self,input:Any,**kwargs):
        if input is None:
            return ""
        
        # output = self.excurtor[0]._call(input,**kwargs)
        if isinstance(input,str):
            output = self.excurtor[1]._call(input,**kwargs)
        else:
            output = self.excurtor[0]._call(input,**kwargs)
        
        return output
    
    async def arun(self,input:Any,**kwargs):
        return self.run(input,**kwargs)
    
    def set_tone(self,path:str):
        self.excurtor[1].set_tone(path)

    

class TranslateTask(Task):
    def init_model(self):
        model = ModelFactory.get_model("translate")
        return [model]

    
class TaskFactory:
    _instances = {}
    _lock = threading.Lock()  # 异步锁

    @staticmethod
    def create_task(task_type) -> Task:
        if task_type not in TaskFactory._instances:
            with TaskFactory._lock:
                if task_type not in TaskFactory._instances:
                    try:
                        if task_type == TASK_AGENT:
                            instance = Agent()
                        elif task_type == TASK_TRANSLATE:
                            instance = TranslateTask()
                        elif task_type == TASK_IMAGE_GEN:
                            instance = ImageGenTask()
                        elif task_type == TASK_SPEECH:
                            instance = Speech()
                        elif task_type == TASK_GENERAL:
                            instance = General()
                        elif task_type == TASK_RETRIEVER:
                            instance = Retriever()

                        TaskFactory._instances[task_type] = instance
                    except Exception as e:
                        print(e)

        return TaskFactory._instances[task_type]

    @staticmethod
    def release():
        with TaskFactory._lock:
            for k,task in TaskFactory._instances.items():
                last_call_time = task.get_last_call_time
                if last_call_time is None:
                    continue

                elapsed_time = time.time() - last_call_time
                elapsed_minutes = int(elapsed_time / 60)
                # print("elapsed_time:",elapsed_time)
                if elapsed_minutes >= 10:
                    model_names = task.bind_model_name()
                    print("ready to release ",model_names)
                    if model_names is not None:
                        task.destroy()
                        for name in model_names:
                            ModelFactory.destroy(name)


        