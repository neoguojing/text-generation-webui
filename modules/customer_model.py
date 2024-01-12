from modules import shared
from modules.callbacks import Iteratorize
from modules.logging_colors import logger
from apps.main import AsyncioThread,input,terminator_output,to_agent,to_speech
from apps.tasks import TaskFactory,TASK_AGENT,TASK_SPEECH,TASK_IMAGE_GEN,TASK_GENERAL,TASK_RETRIEVER
class CustomerModel:
    def __init__(self):
        self.agent = TaskFactory.create_task(TASK_AGENT)
        self.speech = TaskFactory.create_task(TASK_SPEECH)
        self.image_gen = TaskFactory.create_task(TASK_IMAGE_GEN)
        self.general = TaskFactory.create_task(TASK_GENERAL)
        self.retriever = TaskFactory.create_task(TASK_RETRIEVER)
        self.loop = AsyncioThread()
        self.loop.start()

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
    
    def generate(self, prompt, state, callback=None,**kwargs):
        prompt = prompt if type(prompt) is str else prompt.decode()
        output = None

        history = state['history']['internal']
        if len(history) > 5:
            history = history[-5:]

        print("CustomerModel history:",history)
        # print("state:",state)

        texts = self.retriever.retrieve_documents(prompt)
        print("retrieve_documents:",texts)
        if len(texts) > 0:
            history.append([texts[0],""])

        if state['character_menu'].strip() != 'Assistant':
            system = state['context']
            output = self.general.run(prompt,system=system,history=history)
        else:
            # if last input is a image then do image to image task
            if isinstance(prompt,str) and len(history) > 0 and \
                (history[-1][1] is None or history[-1][1] == "") and \
                (self.is_pil_image(history[-1][0]) or self.is_image_path(history[-1][0])):
                image_path = ""
                image_obj = None
                prev_question = history[-1][0] 
                print("prev_question:",type(prev_question))
                if isinstance(prev_question,str):
                    image_path = prev_question
                else:
                    image_obj = prev_question
                output = self.image_gen.run(prompt,image_path=image_path,image_obj=image_obj)

            elif isinstance(prompt,str):
                print("agent input:",prompt)
                output = self.agent.run(prompt,history=history)
                print("agent output:",output)
                print("agent speech output",state['speech_output'])
                if state['speech_output']:
                    output = self.text2audio(output)

        # else:
        #     text_output = self.speech.run(prompt,**kwargs)
        #     analyse_output = self.agent.run(text_output,**kwargs)
        #     audio_output = self.speech.run(analyse_output,**kwargs)
        #     output = audio_output
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

    def is_image_path(self,path):
        import os
        if not isinstance(path,str):
            return False
        
        if not os.path.isabs(path) or not os.path.exists(path):
            return False

        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        extension = os.path.splitext(path)[-1].lower()
        return extension in image_extensions

    def is_pil_image(self,obj):
        from PIL import Image
        return isinstance(obj, Image.Image)
    
    def set_tone(self,path:str):
        self.speech.set_tone(path)
