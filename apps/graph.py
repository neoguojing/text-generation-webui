import os
import sys
import io
from PIL import Image as PILImage
# 获取当前脚本所在的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将当前package的父目录作为顶层package的路径
top_package_path = os.path.abspath(os.path.join(current_dir, ".."))

# 将顶层package路径添加到sys.path
sys.path.insert(0, top_package_path)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langchain_core.output_parsers import StrOutputParser
from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver
from apps.tasks import tools,TaskFactory,TASK_IMAGE_GEN,TASK_SPEECH
import uuid
from langgraph.prebuilt import create_react_agent
from apps.prompt import AgentPromptTemplate,english_traslate_template,agent_prompt
from apps.base import State,CustomHumanMessage
from langchain_core.runnables import  RunnableConfig


class AgentGraph:
    def __init__(self):
        checkpointer = MemorySaver()
        self.llm = ChatOpenAI(model="llama3.1-fp16:latest",base_url="http://localhost:11434/v1/",api_key="xxx",verbose=True)
        self.llm_with_tools = self.llm.bind_tools(tools=tools)
        # prompt = hub.pull("wfh/react-agent-executor")
        # prompt.pretty_print()
        self.translate_chain = english_traslate_template | self.llm 
        self.prompt = agent_prompt
        self.prompt = self.prompt.partial(system_message="You should provide accurate data for the chart_generator to use.")
        self.prompt = self.prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        self.agent_executor = create_react_agent(self.llm, tools, state_modifier=self.prompt)

        self.builder = StateGraph(State)
        
        self.builder.add_node("tranlate", self.translate_chain)
        self.builder.add_node("speech2text", TaskFactory.create_task(TASK_SPEECH))
        self.builder.add_node("text2image", TaskFactory.create_task(TASK_IMAGE_GEN))
        self.builder.add_node("text2speech", TaskFactory.create_task(TASK_SPEECH))
        self.builder.add_node("image2image", TaskFactory.create_task(TASK_IMAGE_GEN))
        self.builder.add_node("agent", self.agent_executor)

        self.builder.add_conditional_edges("tranlate", self.translate_edge_control,
                                           {"image2image": "image2image", "text2image": "text2image"})
        self.builder.add_edge("speech2text", "agent")

        self.builder.add_conditional_edges("agent", self.agent_edge_control, {END: END, "text2speech": "text2speech"})
        self.builder.add_edge("image2image", END)
        self.builder.add_edge("text2image", END)
        self.builder.add_edge("text2speech", END)
        
        self.builder.add_conditional_edges(START, self.routes,
                                           {"tranlate": "tranlate", "speech2text": "speech2text","agent":"agent"})
        # self.graph = self.builder.compile(checkpointer=checkpointer)
        self.graph = self.builder.compile()

    def routes(self,state: State, config: RunnableConfig):
        messages = state["messages"]
        msg_type = state["input_type"]
        if msg_type == "text":
            return "agent"
        elif msg_type == "image":
            return "tranlate"
        elif msg_type == "speech":
            return "speech2text"

        return END
    
    def agent_edge_control(self,state: State):
        if state["need_speech"]:
            return "text2speech"
        return END
    
    def translate_edge_control(self,state: State):
        message = state["messages"][-1]
        if message.media is None:
            return "text2image"
        
        return 'image2image'
    
    def run(self,input):
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        events = self.graph.stream(input, config)
        
        for event in events:
            print(event)
            for value in event.values():
                messages = value.get("messages")
                if messages:
                    if isinstance(messages, list):
                        messages = value["messages"][-1]
                    
                    if messages.content != "":
                        print(
                            "Assistant:",
                            str(messages.content).replace("\n", "\\n")[:50],
                        )
                    
                    if messages.media is not None:
                        pass
    
    def display(self):
        try:
               # Generate the image as a byte stream
            image_data = self.graph.get_graph().draw_mermaid_png()

            # Create a PIL Image object from the byte stream
            image = PILImage.open(io.BytesIO(image_data))

            # Save the image to a file
            image.save("graph.png")
            print("Image saved as output_image.png")
        except Exception:
            # This requires some extra dependencies and is optional
            pass


if __name__ == '__main__':
    input_example = {
        "messages":  [
            CustomHumanMessage(
                content="今天天气怎么样",
            )
        ],
        "input_type": "text",
        "need_speech": False,
        "status": "in_progress",
    }
    
    g = AgentGraph()
    # g.display()
    # resp = g.translate_chain.invoke({"text":"请画一张地球的图片"})
    # resp = g.llm_with_tools.invoke("draw a earth picture")
    # resp = g.agent_executor.invoke(input_example)
    resp = g.run(input_example)
    # print(resp)