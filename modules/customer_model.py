from modules import shared
from modules.callbacks import Iteratorize
from modules.logging_colors import logger
from apps.main import AsyncioThread,input,terminator_output,to_agent,to_speech

class CustomerModel:
    def __init__(self):
        asyncio_thread = AsyncioThread()
        asyncio_thread.start()

    @classmethod
    def from_pretrained(cls, path):
        result = cls()

        return result, result

    def model_type_is_auto(self):
        return shared.args.model_type is None or shared.args.model_type == "Auto" or shared.args.model_type == "None"

    def model_dir(self, path):
        if path.is_file():
            return path.parent

        return path

    def encode(self, string, **kwargs):
        return string

    def decode(self, ids):
        return ids
    
    def generate(self, prompt, state, callback=None):
        prompt = prompt if type(prompt) is str else prompt.decode()
        prompt = to_agent(prompt,"textgen")
        input.put_nowait(prompt)
        output = terminator_output.get()
        return output

    def generate_with_streaming(self, *args, **kwargs):
        with Iteratorize(self.generate, args, kwargs, callback=None) as generator:
            reply = ''
            for token in generator:
                reply += token
                yield reply
