from langchain_openai import ChatOpenAI

from langgraph.graph import END, StateGraph, START
from langchain_core.output_parsers import StrOutputParser
from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver
from .tasks import tools,TaskFactory,TASK_IMAGE_GEN,TASK_SPEECH
import uuid
from langgraph.prebuilt import create_react_agent
from .prompt import AgentPromptTemplate,english_traslate_template
from .base import State,CustomHumanMessage
from langchain_core.runnables import  RunnableConfig

input_example = {
    "messages":  [
        CustomHumanMessage(
            content="你好",
        )
    ],
    "input_type": "text",
    "need_speech": False,
    "status": "in_progress",
}



class AgentGraph:
    def __init__(self):
        self.llm = ChatOpenAI(model="llama3.1-fp16:latest",base_url="http://localhost:11434/v1/",api_key="xxx",verbose=True)
        # prompt = hub.pull("wfh/react-agent-executor")
        # prompt.pretty_print()
        self.translate_chain = english_traslate_template | self.llm | StrOutputParser()
        self.prompt = AgentPromptTemplate()
        self.agent_executor = create_react_agent(self.llm, tools, messages_modifier=self.prompt)

        self.builder = StateGraph(State)
        self.builder.add_node("inputdecide", self.routes)
        self.builder.add_edge(START, "inputdecide")

        self.builder.add_node("tranlate", self.translate_chain)
        self.builder.add_node("speech2text", TaskFactory.create_task(TASK_SPEECH))
        self.builder.add_node("text2image", TaskFactory.create_task(TASK_IMAGE_GEN))
        self.builder.add_node("text2speech", TaskFactory.create_task(TASK_SPEECH))
        self.builder.add_node("image2image", TaskFactory.create_task(TASK_IMAGE_GEN))
        self.builder.add_node("agent", self.agent_executor)
        
        self.builder.add_edge("inputdecide", "tranlate")
        self.builder.add_edge("inputdecide", "speech2text")
        self.builder.add_edge("inputdecide", "agent")

        self.builder.add_edge("tranlate", self.translate_edge_control,{"image2image": "image2image", "text2image": "text2image"})
        self.builder.add_edge("speech2text", "agent")

        self.builder.add_conditional_edges("agent", self.agent_edge_control, {END: END, "text2speech": "text2speech"})
        self.builder.add_edge("image2image", END)
        self.builder.add_edge("text2image", END)
        checkpointer = MemorySaver()
        self.graph = self.builder.compile(checkpointer=checkpointer)

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
            display(Image(self.graph.get_graph().draw_mermaid_png()))
        except Exception:
            # This requires some extra dependencies and is optional
            pass


if __name__ == '__main__':
    g = AgentGraph()
    g.run(input_example)