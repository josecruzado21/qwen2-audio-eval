import os
import sys
current_file_path = os.path.abspath(__file__)
root_path = os.path.abspath(os.path.join(current_file_path, '../../..'))
sys.path.append(root_path)
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import librosa
from datasets import load_dataset
from utils.utils import resample_numpy_audio
script_dir = Path(__file__).resolve()
finetuned_dir_relative = "qwen2audio-ft/v31-20250527-191857"
finetuned_dir = os.path.join(root_path, "src", "04_finetunning","outputs", finetuned_dir_relative)

def qwen2audio_timbre_range_prompt(audio):
    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "audio": audio},
            {"type": "text", "text": "Listen the audio and classify the timbre range in one of 3 numbers: answer 0 for narrow, 1 for moderate and 2 for wide. Answer ONLY with a number"}
        ]},]
    return conversation

def qwen2audio_timbre_range_inference_ft():
    data_type = "train"
    output_path = script_dir.parent / ".." / ".." / "data" / f"fintune_qwen2audio_timbre_range.csv"
    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
        audio_index, predicted_ranges, labels = list(df["audio_index"]), list(df["labels"]), list(df["timbre_range_post_ft"])
    else:
        df=pd.DataFrame(columns=["audio_index", "labels", "timbre_range_post_ft"])
        df.to_csv(output_path, index=False)
        audio_index, predicted_ranges, labels = [], [], []


    timbre_range = load_dataset("ccmusic-database/timbre_range", "range")

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct"")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(finetuned_dir, device_map="auto")
    
    initial_idx = len(predicted_ranges)
    for idx in tqdm(range(initial_idx, 200)):
        waveform = timbre_range[data_type]["audio"][idx]["array"]
        sample_rate = timbre_range[data_type]["audio"][idx]["sampling_rate"]
        label = timbre_range[data_type]["label"][idx]
        audio_index.append(idx)
        waveform_resampled = resample_numpy_audio(waveform, sample_rate)
        conversation = qwen2audio_timbre_range_prompt(waveform_resampled)
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios = [waveform_resampled]
        inputs = processor(text=text, audio=audios, return_tensors="pt", padding=True, sampling_rate=16000)
        # inputs = inputs.to("mps")
        generate_ids = model.generate(**inputs, max_new_tokens=16)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        predicted_ranges.append(response)
        labels.append(label)
        breakpoint()
        df=pd.DataFrame(columns=["audio_index", "labels", "timbre_range_post_ft"])
        df["audio_index"] = audio_index
        df["labels"] = labels
        df["timbre_range_post_ft"] = predicted_ranges
        df.to_csv(output_path, index=False)

if __name__ == "__main__":
    qwen2audio_timbre_range_inference_ft()