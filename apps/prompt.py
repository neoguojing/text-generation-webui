
from langchain.prompts import PromptTemplate,ChatPromptTemplate, HumanMessagePromptTemplate,SystemMessagePromptTemplate
import json
class PromptFactory:
    @staticmethod
    def qa_data_generate_prompt(format_instructions):
        return ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template(
                    "将\n{text}转换中文问答对，格式如下：\n{format_instructions}"
                )
            ],
            input_variables=["text"],
            partial_variables={
                "format_instructions": format_instructions,
            },
        )

    @staticmethod
    def caibao_analyse_prompt(format_instructions):
        template="您是一个专业的财务报表分析师,能够通过用户输入的文本信息，分析有价值的信息，并将分析结果转换为问答形式，输出{format_instructions}格式;请使用中文"
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)

        return ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template(
                    "{text}"
                ),
                system_message_prompt
            ],
            input_variables=["text"],
            partial_variables={
                "format_instructions": format_instructions,
            },
        )


from typing import List, Union
from langchain.prompts import BaseChatPromptTemplate
from langchain.agents import Tool
from langchain_core.messages import HumanMessage, BaseMessage,SystemMessage,AIMessage

TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful \
    for? {description_for_model} Parameters: {parameters} Format the arguments as a JSON object."""

REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tool_descs}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {query}"""

def build_planning_prompt(TOOLS, query):
    tool_descs = []
    tool_names = []
    for info in TOOLS:
        tool_descs.append(
            TOOL_DESC.format(
                name_for_model=info['name_for_model'],
                name_for_human=info['name_for_human'],
                description_for_model=info['description_for_model'],
                parameters=json.dumps(
                    info['parameters'], ensure_ascii=False),
            )
        )
        tool_names.append(info['name_for_model'])
    tool_descs = '\n\n'.join(tool_descs)
    tool_names = ','.join(tool_names)

    prompt = REACT_PROMPT.format(tool_descs=tool_descs, tool_names=tool_names, query=query)
    return prompt

template = """Complete the objective as best you can. You have access to the following tools:

{tools}

Please answer the question based on the following facts. The factual basis for reference is as follows:

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

These were previous tasks you completed:



Begin!

Question: {input}
{agent_scratchpad}"""



import pdb
class QwenAgentPromptTemplate(BaseChatPromptTemplate):
    
    # The template to use
    template: str = template
    # The list of tools available
    tools: List[Tool]

    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")

        # print("intermediate_steps:",intermediate_steps)
        thoughts = ""
        for action, observation in intermediate_steps:
            # print("action:",action)
            # print("observation:",observation)
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [
                HumanMessage(content=formatted)
            ]

sys_agent_template = """Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, 
from answering simple questions to providing in-depth explanations and discussions on a wide range of topics.
As a language model, Assistant is able to generate human-like text based on the input it receives, 
allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. 
It is able to process and understand large amounts of text, 
and can use this knowledge to provide accurate and informative responses to a wide range of questions. 
Additionally, Assistant is able to generate its own text based on the input it receives, 
allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. 
Whether you need help with a specific question or just want to have a conversation about a particular topic, 
Assistant is here to assist.

TOOLS:
------

Assistant has access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

"""

class AgentPromptTemplate(BaseChatPromptTemplate):
    
    # The template to use
    sys_template: str = sys_agent_template
    # The list of tools available
    tools: List[Tool]

    def format_messages(self, **kwargs) -> List[BaseMessage]:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")

        print("intermediate_steps:",intermediate_steps)
        thoughts = ""
        for action, observation in intermediate_steps:
            # print("action:",action)
            # print("observation:",observation)
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.sys_template.format(**kwargs)
        return [
                SystemMessage(content=formatted),
                AIMessage(content=kwargs["agent_scratchpad"]) if "agent_scratchpad" in kwargs else None,
                HumanMessage(content=kwargs["input"]),
            ]
      
def translate_prompt(input_text):
    template = """Translate {input} to English and only return the translation result"""
    prompt = PromptTemplate.from_template(template)
    return prompt.format(input=input_text)

def stock_code_prompt(input_text):
    template = """Stock Symbol or Ticker Symbol of {input}"""
    prompt = PromptTemplate.from_template(template)
    return prompt.format(input=input_text)

def system_prompt(system,context=""):
    template = """{system}\nPlease answer the question based on the following facts. The factual basis for reference is as follows:\n\n{context}"""
    prompt = PromptTemplate.from_template(template)
    return prompt.format(system=system,context=context)


english_traslate_template = ChatPromptTemplate.from_messages(
    [("system", """Translate the following into English and only return the translation result:"""), ("user", "{text}")]
)