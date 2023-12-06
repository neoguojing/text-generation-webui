from modules import shared
from modules.callbacks import Iteratorize
from modules.logging_colors import logger
from apps.main import AsyncioThread,input,terminator_output,to_agent,to_speech

from apps.tasks import TaskFactory,TASK_AGENT,TASK_SPEECH
import pdb
class CustomerModel:
    def __init__(self):
        self.agent = TaskFactory.create_task(TASK_AGENT)
        self.speech = TaskFactory.create_task(TASK_SPEECH)

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
        pdb.set_trace()
        prompt = prompt if type(prompt) is str else prompt.decode()
        output = None
        if isinstance(prompt,str):
            print("--------input:",prompt)
            output = self.agent.run(prompt)
            print("--------output:",output)
        else:
            text_output = self.speech.run(prompt)
            analyse_output = self.agent.run(text_output)
            audio_output = self.speech.run(analyse_output)
            output = audio_output
        return output

    def generate_with_streaming(self, *args, **kwargs):
        with Iteratorize(self.generate, args, kwargs, callback=None) as generator:
            reply = ''
            for token in generator:
                reply += token
                yield reply
