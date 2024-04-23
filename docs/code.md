```
/data/llama/text-generation-webui/modules/chat.py(342)generate_chat_reply_wrapper()
-> for i, history in enumerate(generate_chat_reply(text, state, regenerate, _continue, loading_message=True)):
  /data/llama/text-generation-webui/modules/chat.py(310)generate_chat_reply()
-> for history in chatbot_wrapper(text, state, regenerate=regenerate, _continue=_continue, loading_message=loading_message):
  /data/llama/text-generation-webui/modules/chat.py(247)chatbot_wrapper()
-> for j, reply in enumerate(generate_reply(prompt, state, stopping_strings=stopping_strings, is_chat=True)):
  /data/llama/text-generation-webui/modules/text_generation.py(30)generate_reply()
-> for result in _generate_reply(*args, **kwargs):
  /data/llama/text-generation-webui/modules/text_generation.py(81)_generate_reply()
-> for reply in generate_func(question, original_question, seed, state, stopping_strings, is_chat=is_chat):
> /data/llama/text-generation-webui/modules/text_generation.py(389)generate_reply_custom()

```

langchain:

class Chain.invoke

class AgentExecutor(Chain)._call

LLMSingleActionAgent.plan

class Chain.run
Chain.invoke
class LLMChain(Chain)._call
class LLMChain(Chain).generate
class LLMChain(Chain).prep_prompts
