class AgentExecutor(Chain):
    """Agent that is using tools."""

    agent: Union[BaseSingleActionAgent, BaseMultiActionAgent]
    """The agent to run for creating a plan and determining actions
    to take at each step of the execution loop."""
    tools: Sequence[BaseTool]
    """The valid tools the agent can call."""
    return_intermediate_steps: bool = False
    """Whether to return the agent's trajectory of intermediate steps
    at the end in addition to the final output."""
    max_iterations: Optional[int] = 15
    """The maximum number of steps to take before ending the execution
    loop.
    
    Setting to 'None' could lead to an infinite loop."""
    max_execution_time: Optional[float] = None
    """The maximum amount of wall clock time to spend in the execution
    loop.
    """
    early_stopping_method: str = "force"
    """The method to use for early stopping if the agent never
    returns `AgentFinish`. Either 'force' or 'generate'.

    `"force"` returns a string saying that it stopped because it met a
        time or iteration limit.
    
    `"generate"` calls the agent's LLM Chain one final time to generate
        a final answer based on the previous steps.
    """
    handle_parsing_errors: Union[
        bool, str, Callable[[OutputParserException], str]
    ] = False
    """How to handle errors raised by the agent's output parser.
    Defaults to `False`, which raises the error.
    If `true`, the error will be sent back to the LLM as an observation.
    If a string, the string itself will be sent to the LLM as an observation.
    If a callable function, the function will be called with the exception
     as an argument, and the result of that function will be passed to the agent
      as an observation.
    """
    trim_intermediate_steps: Union[
        int, Callable[[List[Tuple[AgentAction, str]]], List[Tuple[AgentAction, str]]]
    ] = -1

class LLMSingleActionAgent(BaseSingleActionAgent):
    """Base class for single action agents."""

    llm_chain: LLMChain
    """LLMChain to use for agent."""
    output_parser: AgentOutputParser
    """Output parser to use for agent."""
    stop: List[str]
    """List of strings to stop on."""

class LLMChain(Chain):
    """Chain to run queries against LLMs.

    Example:
        .. code-block:: python

            from langchain.chains import LLMChain
            from langchain_community.llms import OpenAI
            from langchain_core.prompts import PromptTemplate
            prompt_template = "Tell me a {adjective} joke"
            prompt = PromptTemplate(
                input_variables=["adjective"], template=prompt_template
            )
            llm = LLMChain(llm=OpenAI(), prompt=prompt)
    """

[docs]    @classmethod
    def is_lc_serializable(self) -> bool:
        return True


    prompt: BasePromptTemplate
    """Prompt object to use."""
    llm: Union[
        Runnable[LanguageModelInput, str], Runnable[LanguageModelInput, BaseMessage]
    ]
    """Language model to call."""
    output_key: str = "text"  #: :meta private:
    output_parser: BaseLLMOutputParser = Field(default_factory=StrOutputParser)
    """Output parser to use.
    Defaults to one that takes the most likely string but does not change it 
    otherwise."""
    return_final_only: bool = True
    """Whether to return only the final parsed result. Defaults to True.
    If false, will return a bunch of extra information about the generation."""
    llm_kwargs: dict = Field(default_factory=dict)


agent 流程：

1. excutor.invoke 实际调用的是第二步
2. Chain.invoke： 
a. 配置 callback_manager,调用on_chain_start
b. 通过input_keys，校验是否缺少输入参数
c. 调用_call函数,获取输出
d. 组装输出，支持输出input和output，也支持单独输出output，返回格式Dict[str, Any]
e: 调用on_chain_end
3. AgentExecutor._call：最终返回是Dict[str, Any]
a._should_continue: 通过max_execution_time或max_iterations 控制执行次数
循环：
    b. _take_next_step->_consume_next_step->_aiter_next_step：最终返回Union[AgentFinish, List[Tuple[AgentAction, str]]]
    c. 若输出是AgentFinish，则调用_return返回，返回结果是Dict[str, Any]
    d. 否则保存中间结果，中间结果为一个的情况下，调用_get_tool_return来判断tool是否支持直接返回
    e. self.agent.return_stopped_response:处理强制退出的返回 force: 因为超时或者超过循环次数  generate：从intermediate_steps中组织输入和输出，交给大模型做最后总结，有outputparser则调用
4. 循环执行：AgentExecutor._take_next_step
5. RunnableAgent(BaseSingleActionAgent).plan：返回action或者finish
   a 调用self.runnable.invoke 或者self.runnable.stream
6. RunnableSequence._stream
7. RunnableSequence.transform
8. _transform_stream_with_config


_consume_next_step： 负责过滤agent的所有结果，若最后一个是finish则只返回一个结果，否则将AgentStep 分解为action和observation
_iter_next_step： 裁剪intermediate_steps，调用agent.plan，finish则返回，action遍历返回，函数则调用_perform_agent_action
_perform_agent_action：输入action调用tool.run ,返回AgentStep

AgentAction.message ： 将action转换为AIMessage
AgentExecutor 继承了 Chain
input_keys 函数的实现：
1. LLMChain：  self.prompt.input_variables
2.Agent:list(set(self.llm_chain.input_keys) - {"agent_scratchpad"})

BaseTool： return_direct 控制tool的结果是否直接返回；tool继承该函数并实现run接口
handle_parsing_errors： 可以是bool 函数或者字符串
class AgentStep(Serializable):
    """The result of running an AgentAction."""

    action: AgentAction
    """The AgentAction that was executed."""
    observation: Any
    """The result of the AgentAction."""

class AgentAction(Serializable):
    """A full description of an action for an ActionAgent to execute."""

    tool: str
    """The name of the Tool to execute."""
    tool_input: Union[str, dict]
    """The input to pass in to the Tool."""
    log: str

class AgentFinish(Serializable):
    """The final return value of an ActionAgent."""

    return_values: dict
    """Dictionary of return values."""
    log: str