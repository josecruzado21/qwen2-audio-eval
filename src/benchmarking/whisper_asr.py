import torchaudio
import os
import whisper
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
script_dir = Path(__file__).resolve()


def whisper_inference():
    data_type = "dev-clean"
    output_path = script_dir.parent / ".." / ".." / "data" / f"librispeech_whisper_{data_type}.csv"

    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
        original_text, transcripts = list(df["original_text"]), list(df["transcripts"])
    else:
        df=pd.DataFrame(columns=["original_text", "transcripts"])
        df.to_csv(output_path, index=False)
        original_text, transcripts = [], []

    # Load Librispeech
    librispeech = torchaudio.datasets.LIBRISPEECH(
            root = "/share/data/lang/users/ttic_31110/jcruzado/data/librispeech_dev_clean", 
            url=data_type, 
            download=False)
    model = whisper.load_model("large", download_root =  "/share/data/lang/users/ttic_31110/jcruzado/models/whisper")

    initial_idx = len(transcripts)
    for idx in tqdm(range(initial_idx, 200)):
        waveform, sample_rate, transcript, *_ = librispeech[idx]
        original_text.append(transcript)
        waveform_np = waveform.squeeze().numpy().astype(np.float32)
        result = model.transcribe(waveform_np, language="en")
        transcripts.append(result["text"])
        breakpoint()
        df["original_text"] = original_text
        df["transcripts"] = transcripts
        df.to_csv(output_path, index=False)

if __name__ == "__main__":
    whisper_inference()


