import os
import re
from datetime import datetime
from pathlib import Path

from modules import github, shared
from modules.logging_colors import logger


# Helper function to get multiple values from shared.gradio
def gradio(*keys):
    if len(keys) == 1 and type(keys[0]) in [list, tuple]:
        keys = keys[0]

    return [shared.gradio[k] for k in keys]


def save_file(fname, contents):
    if fname == '':
        logger.error('File name is empty!')
        return

    root_folder = Path(__file__).resolve().parent.parent
    abs_path = Path(fname).resolve()
    rel_path = abs_path.relative_to(root_folder)
    if rel_path.parts[0] == '..':
        logger.error(f'Invalid file path: {fname}')
        return

    with open(abs_path, 'w', encoding='utf-8') as f:
        f.write(contents)

    logger.info(f'Saved {abs_path}.')


def delete_file(fname):
    if fname == '':
        logger.error('File name is empty!')
        return

    root_folder = Path(__file__).resolve().parent.parent
    abs_path = Path(fname).resolve()
    rel_path = abs_path.relative_to(root_folder)
    if rel_path.parts[0] == '..':
        logger.error(f'Invalid file path: {fname}')
        return

    if abs_path.exists():
        abs_path.unlink()
        logger.info(f'Deleted {fname}.')


def current_time():
    return f"{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"


def atoi(text):
    return int(text) if text.isdigit() else text.lower()


# Replace multiple string pairs in a string
def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)

    return text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def get_available_models():
    model_list = []
    for item in list(Path(f'{shared.args.model_dir}/').glob('*')):
        if not item.name.endswith(('.txt', '-np', '.pt', '.json', '.yaml', '.py')) and 'llama-tokenizer' not in item.name:
            model_list.append(re.sub('.pth$', '', item.name))

    return ['None'] + sorted(model_list, key=natural_keys)


def get_available_presets():
    return sorted(set((k.stem for k in Path('presets').glob('*.yaml'))), key=natural_keys)


def get_available_prompts():
    prompts = []
    files = set((k.stem for k in Path('prompts').glob('*.txt')))
    prompts += sorted([k for k in files if re.match('^[0-9]', k)], key=natural_keys, reverse=True)
    prompts += sorted([k for k in files if re.match('^[^0-9]', k)], key=natural_keys)
    prompts += ['None']
    return prompts


def get_available_characters():
    paths = (x for x in Path('characters').iterdir() if x.suffix in ('.json', '.yaml', '.yml'))
    return sorted(set((k.stem for k in paths)), key=natural_keys)


def get_available_instruction_templates():
    path = "instruction-templates"
    paths = []
    if os.path.exists(path):
        paths = (x for x in Path(path).iterdir() if x.suffix in ('.json', '.yaml', '.yml'))

    return ['None'] + sorted(set((k.stem for k in paths)), key=natural_keys)


def get_available_extensions():
    extensions = sorted(set(map(lambda x: x.parts[1], Path('extensions').glob('*/script.py'))), key=natural_keys)
    extensions = [v for v in extensions if v not in github.new_extensions]
    return extensions


def get_available_loras():
    return ['None'] + sorted([item.name for item in list(Path(shared.args.lora_dir).glob('*')) if not item.name.endswith(('.txt', '-np', '.pt', '.json'))], key=natural_keys)


def get_datasets(path: str, ext: str):
    # include subdirectories for raw txt files to allow training from a subdirectory of txt files
    if ext == "txt":
        return ['None'] + sorted(set([k.stem for k in list(Path(path).glob('*.txt')) + list(Path(path).glob('*/')) if k.stem != 'put-trainer-datasets-here']), key=natural_keys)

    return ['None'] + sorted(set([k.stem for k in Path(path).glob(f'*.{ext}') if k.stem != 'put-trainer-datasets-here']), key=natural_keys)


def get_available_chat_styles():
    return sorted(set(('-'.join(k.stem.split('-')[1:]) for k in Path('css').glob('chat_style*.css'))), key=natural_keys)


def get_available_grammars():
    return ['None'] + sorted([item.name for item in list(Path('grammars').glob('*.gbnf'))], key=natural_keys)

from datetime import date
from scipy.io import wavfile
import time
def audio_save(audio_data,sample_rate,file_path="./audio/input"):
    import numpy as np
    file = f'{date.today().strftime("%Y_%m_%d")}/{int(time.time())}'  # noqa: E501
    output_file = Path(f"{file_path}/{file}.wav")
    output_file.parent.mkdir(parents=True, exist_ok=True)
        
    wavfile.write(output_file,rate=sample_rate, data=audio_data)

    formatted_result = f'<audio controls src="file/{output_file}">'
    formatted_result += 'Your browser does not support the audio element.'
    formatted_result += '</audio>'
    return formatted_result,output_file

def image_save(image_obj,file_path="./pics/input"):
    file = f'{date.today().strftime("%Y_%m_%d")}/{int(time.time())}'  # noqa: E501
    output_file = Path(f"{file_path}/{file}.png")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    style = 'style="width: 100%; max-height: 100vh;"'
    image_obj.save(output_file)
    formatted_result = f'<img src="file/{output_file}" {style}>'
    return formatted_result,output_file

def down_sampling(data,orig_sr,target_sr=16000):
    import scipy.signal as sps
    # Resample data
    number_of_samples = round(len(data) * float(target_sr) / orig_sr)
    decimated_data = sps.resample(data, number_of_samples)
    # downsample_factor = int(orig_sr / target_sr)  # 降低到目标采样率16,000 Hz
    # decimated_data = sps.decimate(data, downsample_factor)
    return decimated_data