from langchain_openai import ChatOpenAI

from langgraph.graph.message import AnyMessage, add_messages,AIMessage
from langgraph.graph import END, StateGraph, START
from langgraph.graph import END, StateGraph, START
from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver
from .tasks import tools,TaskFactory,TASK_IMAGE_GEN,TASK_SPEECH
import uuid
from langgraph.prebuilt import create_react_agent
from .prompt import AgentPromptTemplate
from .base import State

# silver_input = {
#     "messages": [("user", silver_row["description"])],
#     "test_cases": silver_row["test_cases"],
#     "runtime_limit": silver_row["runtime_limit"],
#     "status": "in_progress",
# }



class AgentGraph:
    def __init__(self):
        self.llm = ChatOpenAI(model="llama3.1-fp16:latest",base_url="http://localhost:11434/v1/",api_key="xxx",verbose=True)
        # prompt = hub.pull("wfh/react-agent-executor")
        # prompt.pretty_print()
        self.prompt = AgentPromptTemplate()
        self.agent_executor = create_react_agent(self.llm, tools, messages_modifier=self.prompt)
        self.builder = StateGraph(State)
        self.builder.add_node("inputdecide", solver)
        self.builder.add_edge(START, "inputdecide")
        self.builder.add_node("tranlate", solver)
        self.builder.add_node("speech2text", TaskFactory.create_task(TASK_SPEECH))
        self.builder.add_node("text2image", TaskFactory.create_task(TASK_IMAGE_GEN))
        self.builder.add_node("text2speech", TaskFactory.create_task(TASK_SPEECH))
        self.builder.add_node("image2image", TaskFactory.create_task(TASK_IMAGE_GEN))
        self.builder.add_node("agent", self.agent_executor)
        self.builder.add_edge("inputdecide", "tranlate")
        self.builder.add_edge("inputdecide", "speech2text")
        self.builder.add_edge("inputdecide", "image2image")
        self.builder.add_edge("tranlate", "text2image")
        self.builder.add_edge("speech2text", "agent")
        self.builder.add_edge("inputdecide", "agent")
        self.builder.add_edge("agent", "text2speech")


        self.builder.add_conditional_edges("evaluate", self.control_edge, {END: END, "solver": "solver"})
        self.builder.add_edge("text2speech", END)
        self.builder.add_edge("agent", END)
        self.builder.add_edge("image2image", END)
        self.builder.add_edge("text2image", END)
        checkpointer = MemorySaver()
        self.graph = self.builder.compile(checkpointer=checkpointer)

    def control_edge(self,state: State):
        if state.get("status") == "success":
            return END
        return "solver"
    
    def run(self,input):
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        events = self.graph.stream(input, config)
        for event in events:
            for value in event.values():
                messages = value.get("messages")
                if messages:
                    if isinstance(messages, list):
                        messages = value["messages"][-1]
                    print(
                        "Assistant:",
                        str(messages.content).replace("\n", "\\n")[:50],
                    )
                elif value.get("examples"):
                    print("Retrieved examples:\n\n", value["examples"][:100] + "...")
                elif value.get("candidate"):
                    print(str(value["candidate"].content)[:200])


    def routes(self,state: State, name: str = "Subject_Matter_Expert"):
        messages = state["messages"]
        num_responses = len(
            [m for m in messages if isinstance(m, AIMessage) and m.name == name]
        )
        if num_responses >= max_num_turns:
            return END
        last_question = messages[-2]
        if last_question.content.endswith("Thank you so much for your help!"):
            return END
        return "ask_question"
    
    def display(self):
        try:
            display(Image(self.graph.get_graph().draw_mermaid_png()))
        except Exception:
            # This requires some extra dependencies and is optional
            pass


if __name__ == '__main__':
    pass