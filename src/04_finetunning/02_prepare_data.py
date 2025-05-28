# prepare_dataset.py
import os
import json
from tqdm import tqdm
from datasets import load_dataset
import soundfile as sf
import sys

# Set up paths
current_file_path = os.path.abspath(__file__)
root_path = os.path.abspath(os.path.join(current_file_path, '../../..'))
sys.path.append(root_path)
from utils.utils import resample_numpy_audio
data_dir = os.path.join(root_path, "data", "finetunning_data")
audio_dir = os.path.join(data_dir, "data", "audio_clips")
os.makedirs(audio_dir, exist_ok=True)

# Load dataset
print("Loading timbre_range dataset...")
ds = load_dataset("ccmusic-database/timbre_range", "range", split="train")

# Use only the first 200 examples
ds = ds.select(range(200))

# Instruction
instruction = "Listen the audio and classify the timbre range: answer 0 for narrow, 1 for moderate and 2 for wide. Answer only with a number"

# Build JSON
output = []
for i in tqdm(range(len(ds)), desc="Processing train subset"):
    example = ds[i]
    label = str(example["label"])
    waveform = example["audio"]["array"]
    sr = example["audio"]["sampling_rate"]

    # Resample to 16kHz
    waveform_resampled = resample_numpy_audio(waveform, sr, target_sr=16000)

    # Save to .wav
    wav_filename = f"train_{i:04d}.wav"
    wav_path = os.path.join(audio_dir, wav_filename)
    sf.write(wav_path, waveform_resampled, 16000)

    # Swift-compatible conversation format
    item = {
        "conversations": [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": wav_path},
                    {"type": "text", "text": instruction}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": label}]
            }
        ]
    }
    output.append(item)

# Save JSON
train_json_path = os.path.join(data_dir, "train.json")
with open(train_json_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"✅ Done. train.json saved with {len(output)} entries at {train_json_path}")
