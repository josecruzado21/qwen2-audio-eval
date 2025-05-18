import os
import sys
current_file_path = os.path.abspath(__file__)
root_path = os.path.abspath(os.path.join(current_file_path, '../../..'))
sys.path.append(root_path)
from urllib.request import urlopen
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import librosa
script_dir = Path(__file__).resolve()

def qwen2audio_audioonly_chat_prompt(row, audio_filename):
    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "audio_url": audio_filename},
        ]},]
    return conversation

def qwen2audio_audioonly_inference(MMLU_data):
    output_path = script_dir.parent / ".." / ".." / "data" / "MMLU.csv"
    col_name = "qwen2audio_audioonly_response"
    if col_name not in MMLU_data.columns:
        MMLU_data[col_name] = ""
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", 
                                              cache_dir = "/share/data/lang/users/ttic_31110/jcruzado/models/")
    model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto", 
                                                              cache_dir = "/share/data/lang/users/ttic_31110/jcruzado/models/")

    for idx, row in tqdm(MMLU_data.iterrows()):
        if (row[col_name] == "") | pd.isna(row[col_name]):
            conversation = qwen2audio_audioonly_chat_prompt(row, row["speech_file"])
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios = []
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if ele["type"] == "audio":
                            waveform, _ = librosa.load(ele["audio_url"], sr=processor.feature_extractor.sampling_rate)
                            audios.append(waveform)
            inputs = processor(text=text, audio=audios, return_tensors="pt", padding=True)
            inputs = inputs.to("cuda")
            generate_ids = model.generate(**inputs, max_new_tokens=8)
            generate_ids = generate_ids[:, inputs.input_ids.size(1):]
            response = processor.batch_decode(generate_ids, skip_special_tokens=True, 
                                              clean_up_tokenization_spaces=False)[0]
            MMLU_data.at[idx, col_name] = response
            MMLU_data.to_csv(output_path, index=False) 

if __name__ == "__main__":
    MMLU_data = pd.read_csv(os.path.join(root_path, "data", "MMLU.csv"))
    qwen2audio_audioonly_inference(MMLU_data)