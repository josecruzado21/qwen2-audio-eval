import os
import torchaudio
import sys
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
script_dir = Path(__file__).resolve()

def qwen2audio_asr_inference():
    data_type = "dev-clean"
    output_path = script_dir.parent / ".." / ".." / "data" / f"librispeech_qwen2audio_{data_type}.csv"
    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
        original_text, transcripts = list(df["original_text"]), list(df["transcripts"])
    else:
        df=pd.DataFrame(columns=["original_text", "transcripts"])
        df.to_csv(output_path, index=False)
        original_text, transcripts = [], []


    librispeech = torchaudio.datasets.LIBRISPEECH(
            root = "/share/data/lang/users/ttic_31110/jcruzado/data/librispeech_dev_clean", 
            url=data_type, 
            download=False)

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B", 
                                              cache_dir = "/share/data/lang/users/ttic_31110/jcruzado/models/")
    model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B", device_map="auto", 
                                                               cache_dir = "/share/data/lang/users/ttic_31110/jcruzado/models/")
    
    initial_idx = len(transcripts)
    for idx in tqdm(range(initial_idx, 200)):
        waveform, sample_rate, transcript, *_ = librispeech[idx]
        original_text.append(transcript)
        waveform_np = waveform.squeeze().numpy().astype(np.float32)
        inputs = processor(text="<|audio_bos|><|AUDIO|><|audio_eos|>Transcribe the audio:", audio=[waveform_np], sampling_rate=sample_rate, 
                           return_tensors="pt", padding=True)
        inputs = inputs.to("cuda")
        generate_ids = model.generate(**inputs, max_new_tokens=256)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        transcripts.append(response)
        df["original_text"] = original_text
        df["transcripts"] = transcripts
        df.to_csv(output_path, index=False)

if __name__ == "__main__":
    qwen2audio_asr_inference()