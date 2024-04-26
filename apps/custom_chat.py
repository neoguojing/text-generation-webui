from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, SimpleChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage,AIMessage,SystemMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult,LLMResult
from langchain_core.runnables import run_in_executor
import pdb


class CustomChatModelAdvanced(BaseChatModel):
    """A custom chat model that echoes the first `n` characters of the input.

    When contributing an implementation to LangChain, carefully document
    the model including the initialization parameters, include
    an example of how to initialize the model and include any relevant
    links to the underlying models documentation or API.

    Example:

        .. code-block:: python

            model = CustomChatModel(n=2)
            result = model.invoke([HumanMessage(content="hello")])
            result = model.batch([[HumanMessage(content="hello")],
                                 [HumanMessage(content="world")]])
    """

    model_name: str
    """The name of the model"""
    n: int
    """The number of characters from the last message of the prompt to be echoed."""

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Override the _generate method to implement the chat model logic.

        This can be a call to an API, a call to a local model, or any other
        implementation that generates a response to the input prompt.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """
        # Replace this with actual logic to generate a response from a list
        # of messages.
        # pdb.set_trace()
        last_message = messages[-1]
        input = self._format_message(messages)
        print(input)
        message = AIMessage(
            content=last_message.content,
            additional_kwargs={},  # Used to add additional payload (e.g., function calling request)
            response_metadata={  # Use for response metadata
                "time_in_seconds": 3,
            },
        )

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

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

    # def _stream(
    #     self,
    #     messages: List[BaseMessage],
    #     stop: Optional[List[str]] = None,
    #     run_manager: Optional[CallbackManagerForLLMRun] = None,
    #     **kwargs: Any,
    # ) -> Iterator[ChatGenerationChunk]:
    #     """Stream the output of the model.

    #     This method should be implemented if the model can generate output
    #     in a streaming fashion. If the model does not support streaming,
    #     do not implement it. In that case streaming requests will be automatically
    #     handled by the _generate method.

    #     Args:
    #         messages: the prompt composed of a list of messages.
    #         stop: a list of strings on which the model should stop generating.
    #               If generation stops due to a stop token, the stop token itself
    #               SHOULD BE INCLUDED as part of the output. This is not enforced
    #               across models right now, but it's a good practice to follow since
    #               it makes it much easier to parse the output of the model
    #               downstream and understand why generation stopped.
    #         run_manager: A run manager with callbacks for the LLM.
    #     """
    #     last_message = messages[-1]
    #     tokens = last_message.content[: self.n]

    #     for token in tokens:
    #         chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))

    #         if run_manager:
    #             # This is optional in newer versions of LangChain
    #             # The on_llm_new_token will be called automatically
    #             run_manager.on_llm_new_token(token, chunk=chunk)

    #         yield chunk

    #     # Let's add some other information (e.g., response metadata)
    #     chunk = ChatGenerationChunk(
    #         message=AIMessageChunk(content="", response_metadata={"time_in_sec": 3})
    #     )
    #     if run_manager:
    #         # This is optional in newer versions of LangChain
    #         # The on_llm_new_token will be called automatically
    #         run_manager.on_llm_new_token(token, chunk=chunk)
    #     yield chunk

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "echoing-chat-model-advanced"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": self.model_name,
        }
    
from langchain.tools import BaseTool, StructuredTool, tool
@tool
def search(query: str) -> str:
    """Look up things online."""
    return "LangChain"

if __name__ == '__main__':
    model = CustomChatModelAdvanced(n=3, model_name="my_custom_model")
    # out = model.invoke(
    #     [
    #         HumanMessage(content="hello!"),
    #         AIMessage(content="Hi there human!"),
    #         HumanMessage(content="Meow!"),
    #     ]
    
    # )
    # print("llm out:",out)
    # from langchain_core.prompts import ChatPromptTemplate
    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", "You are a world class technical documentation writer."),
    #     ("user", "{input}")
    # ])

    # chain = prompt | model 
    # out = chain.invoke({"input": "how can langsmith help with testing?"})
    # print("chain out:",out)
    import os
    import sys
    from typing import Any
    # 获取当前脚本所在的目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 将当前package的父目录作为顶层package的路径
    top_package_path = os.path.abspath(os.path.join(current_dir, ".."))

    # 将顶层package路径添加到sys.path
    sys.path.insert(0, top_package_path)
    from apps.prompt import AgentPromptTemplate
    from apps.parser import QwenAgentOutputParser
    from langchain.chains.llm import LLMChain
    from langchain.agents import AgentExecutor,create_react_agent

    prompt = AgentPromptTemplate(
            tools=[],
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables=["input", "intermediate_steps",'tools', 'tool_names', 'agent_scratchpad']
        )
    
    tools = [search]
    # tool_names = [tool.name for tool in tools]

    agent = create_react_agent(
        llm=model,
        tools=tools,
        prompt=prompt
    )

    excutor = AgentExecutor.from_agent_and_tools(agent=agent,tools=tools, verbose=True)
    pdb.set_trace()
    excutor.invoke({"input": "how can langsmith help with testing?"})
    