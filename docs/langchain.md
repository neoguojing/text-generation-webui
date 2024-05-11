
# agent 流程：

1. **excutor.invoke** 
   - 实际调用的是第二步
2. **Chain.invoke**  输入：Dict[str, Any]，输出：Dict[str, Any]
    - a. 配置 callback_manager,调用on_chain_start
    - b. 通过input_keys，校验是否缺少输入参数
    - c. 调用_call函数,获取输出
    - d. 组装输出，支持输出input和output，也支持单独输出output
    - e: 调用on_chain_end
3. **AgentExecutor._call** 输入：Dict[str, str]，输出：Dict[str, Any]
    - a._**should_continue** 通过max_execution_time或max_iterations 控制执行次数
    - 循环：
       - b. _take_next_step->_consume_next_step->_aiter_next_step
       - c. 若输出是AgentFinish，则调用_return返回，返回结果是Dict[str, Any]
       - d. 否则保存中间结果，中间结果为一个的情况下，调用_get_tool_return来判断tool是否支持直接返回
       - e. self.agent.return_stopped_response:处理强制退出的返回 force: 因为超时或者超过循环次数  generate：从intermediate_steps中组织输入和输出，交给大模型做最后总结，有outputparser则调用
4. **循环执行：AgentExecutor._take_next_step** 输入：Dict[str, str]，输出：Union[AgentFinish, List[Tuple[AgentAction, str]]]
   - 1. **RunnableAgent(BaseSingleActionAgent).plan**： 输出:Union[AgentAction, AgentFinish]
      - a 调用self.runnable.invoke 或者self.runnable.stream
   - 2. **RunnableSequence._stream 或者 RunnableSequence.invoke** : 执行agent调用链，以下步骤均属于该调用链，返回AgentAction
     - 3. **RunnablePassthrough.invoke**: 将intermediate_steps 转换为 agent_scratchpad
     - 4. **BasePromptTemplate.invoke** : 输入：？？，输出：PromptValue (StringPromptValue,ChatPromptValue)
       - BasePromptTemplate._format_prompt_with_error_handling
       -  BaseChatPromptTemplate.format_prompt
       -  自定义模版类.format_messages
     - 5. **RunnableBindingBase.invoke**：调用模型的地方
        ```class RunnableBindingBase(RunnableSerializable[Input, Output]):
            bound: Runnable[Input, Output] 该参数即自定义的llm
        ```
        - 6.  **BaseChatModel.invoke**:返回AIMessage 输出：BaseMessage
        - 7.  **BaseChatModel.generate_prompt**: 输入：List[PromptValue]，输出：LLMResult
        - 将[StringPromptValue] 转换为[HumanMessage]
        - 8.  **BaseChatModel.generate**: 输入： List[List[BaseMessage]]，输出LLMResult
        - 遍历messages，依次调用_generate_with_cache,
        - 9.  **BaseChatModel._generate_with_cache**: 实现一个缓存，缓存相同的请求结果，调用自定义llm的_generate,返回ChatResult
        - 10. **Llama3Chat._generate**：输入:List[BaseMessage],输出：ChatResult
     - 11. **BaseOutputParser.invoke**:执行结果解析，返回AgentAction
       - 12. **BaseOutputParser.parse_result**
       - 13. **ReActSingleInputOutputParser.parse**：输入：str，输出： Union[AgentAction, AgentFinish]
## 重要函数说明
    - AgentExecutor._consume_next_step： 负责过滤agent的所有结果，若最后一个是finish则只返回一个结果，否则将AgentStep 分解为action和observation
    - AgentExecutor._iter_next_step： 裁剪intermediate_steps，调用agent.plan，finish则返回，action遍历返回，函数则调用_perform_agent_action
    - AgentExecutor._perform_agent_action：输入action调用tool.run ,返回AgentStep
    - AgentAction.message ： 将action转换为AIMessage
    - AgentExecutor 继承了 Chain
    - input_keys 函数的实现：
        - 1. LLMChain：  self.prompt.input_variables
        - 2.Agent:list(set(self.llm_chain.input_keys) - {"agent_scratchpad"})

    - BaseTool： return_direct 控制tool的结果是否直接返回；tool继承该函数并实现run接口
    - handle_parsing_errors： 可以是bool 函数或者字符串

## 重要概念：
- intermediate_steps: List[Tuple[AgentAction, str]]:
  - a 在AgentExecutor._call中建立一个空的intermediate_steps
  - b RunnableAgent(BaseSingleActionAgent).plan中将intermediate_steps放入input，作为一个入参
  - c 在在AgentExecutor._call中，将以上步骤中输出的AgentAction和str放到a步骤的数组中
- agent_scratchpad： 由intermediate_steps生成的字符串
  ```
  for action, observation in intermediate_steps:
            # print("action:",action)
            # print("observation:",observation)
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
  kwargs["agent_scratchpad"] = thoughts
  ```
## 重要class
```
    class LLMResult(BaseModel):
        """Class that contains all results for a batched LLM call."""

        generations: List[List[Generation]]
        """List of generated outputs. This is a List[List[]] because
        each input could have multiple candidate generations."""
        llm_output: Optional[dict] = None
        """Arbitrary LLM provider-specific output."""
        run: Optional[List[RunInfo]] = None

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

    class RunnableSequence(RunnableSerializable[Input, Output]):
        first: Runnable[Input, Any]
        """The first runnable in the sequence."""
        middle: List[Runnable[Any, Any]] = Field(default_factory=list)
        """The middle runnables in the sequence."""
        last: Runnable[Any, Output]
        """The last runnable in the sequence."""

    class ChatGeneration(Generation):
        """A single chat generation output."""

        text: str = ""
        """*SHOULD NOT BE SET DIRECTLY* The text contents of the output message."""
        message: BaseMessage
        """The message output by the chat model."""
    
    class StringPromptValue(PromptValue):
    """String prompt value."""
    text: str
    """Prompt text."""

    def to_string(self) -> str:
        """Return prompt as string."""
        return self.text

    def to_messages(self) -> List[BaseMessage]:
        """Return prompt as messages."""
        return [HumanMessage(content=self.text)]


    class ChatPromptValue(PromptValue):
        """Chat prompt value.

        A type of a prompt value that is built from messages.
        """

        messages: Sequence[BaseMessage]
        """List of messages."""

        def to_string(self) -> str:
            """Return prompt as string."""
            return get_buffer_string(self.messages)

        def to_messages(self) -> List[BaseMessage]:
            """Return prompt as a list of messages."""
            return list(self.messages)

```

## 问题
    大模型返回值有问题：多了Observation
    今天上海的天气\nObservation'

