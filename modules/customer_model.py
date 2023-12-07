from modules import shared
from modules.callbacks import Iteratorize
from modules.logging_colors import logger
from apps.main import AsyncioThread,input,terminator_output,to_agent,to_speech

from apps.tasks import TaskFactory,TASK_AGENT,TASK_SPEECH
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
        ids = self.agent.encode(string)
        return ids

    def decode(self, ids):
        string = self.agent.decode(ids)
        return string
    
    def generate(self, prompt, state, callback=None):
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
    
    def audio2text(self, audio_data):
        text_output = self.speech.run(audio_data)
        return text_output
    
    def text2audio(self, text):
        audio_output = self.speech.run(text)
        return audio_output

    def generate_with_streaming(self, *args, **kwargs):
        with Iteratorize(self.generate, args, kwargs, callback=None) as generator:
            reply = ''
            for token in generator:
                reply += token
                yield reply
