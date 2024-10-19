import torch
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))

# 将当前package的父目录作为顶层package的路径
top_package_path = os.path.abspath(os.path.join(current_dir, "../../"))
# 将顶层package路径添加到sys.path
sys.path.insert(0, top_package_path)
from TTS.api import TTS
from apps.config import model_root

class TextToSpeech:
    def __init__(self, model_path: str = os.path.join(model_root,"tts_models--zh-CN--baker--tacotron2-DDC-GST"), 
                 speaker_wav: str = os.path.join(model_root,"XTTS-v2","samples/zh-cn-sample.wav"), 
                 language: str = "zh-cn"):
        self.device = "gpu" if torch.cuda.is_available() else "cpu"
        config_path = os.path.join(model_path,"config.json")
        self.tts = TTS(model_path=model_path,config_path=config_path).to(self.device)
        # self.tts = TTS(model_name=model_path).to(self.device)
        self.speaker_wav = speaker_wav
        self.language = language

    def list_available_models(self):
        return TTS().list_models()

    def generate_audio(self, text: str):
        return self.tts.tts(text=text, speaker_wav=self.speaker_wav, language=self.language)

    def save_audio_to_file(self, text: str, file_path: str):
        # self.tts.tts_to_file(text=text, speaker_wav=self.speaker_wav, language=self.language, file_path=file_path)
        self.tts.tts_to_file(text=text, speaker_wav=self.speaker_wav, file_path=file_path)

# 使用示例
if __name__ == "__main__":
    # tts_instance = TextToSpeech("tts_models/multilingual/multi-dataset/xtts_v2")
    # tts_instance = TextToSpeech()
    tts_instance = TextToSpeech(language="zh-CN")
    # 列出可用模型
    print(tts_instance.list_available_models())
    
    input = '''
        以下是每个缩写的简要解释：

hag: Hanga — 指的是一种语言，主要在巴布亚新几内亚的Hanga地区使用。

hnn: Hanunoo — 指的是菲律宾的一种语言，主要由Hanunoo人使用，属于马来-波利尼西亚语系。

bgc: Haryanvi — 指的是印度哈里亚纳邦的一种方言，属于印地语的一种变体。

had: Hatam — 指的是巴布亚新几内亚的一种语言，主要在Hatam地区使用。

hau: Hausa — 指的是西非的一种语言，广泛用于尼日利亚和尼日尔，是主要的交易语言之一。

hwc: Hawaii Pidgin — 指的是夏威夷的一种克里奥尔语，受英语和夏威夷土著语言影响，常用于当地的日常交流。

hvn: Hawu — 指的是印度尼西亚的一种语言，主要在西努沙登加拉省的Hawu地区使用。

hay: Haya — 指的是坦桑尼亚的一种语言，由Haya人使用，属于尼日尔-刚果语系。
    '''
    # 生成音频并保存到文件
    tts_instance.save_audio_to_file(input, "1tacoutput.wav")
