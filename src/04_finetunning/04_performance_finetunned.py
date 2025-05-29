import os
import sys
current_file_path = os.path.abspath(__file__)
root_path = os.path.abspath(os.path.join(current_file_path, '../../..'))
sys.path.append(root_path)
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from swift.llm import PtEngine
from swift.llm import InferRequest, RequestConfig
script_dir = Path(__file__).resolve()
finetuned_dir_relative = "qwen2audio-ft/v31-20250527-191857"
finetuned_dir = os.path.join(root_path, "src", "04_finetunning","outputs", finetuned_dir_relative)

adapter_path =  "/Users/jose_cruzado/Library/CloudStorage/OneDrive-TheUniversityofChicago/UChicago/MS Statistics/Courses/Spring 2025/Speech Technologies/Project/qwen2-audio-eval/src/04_finetunning/outputs/qwen2audio-ft/v31-20250527-191857/checkpoint-37"
original_model_path = "/Users/jose_cruzado/.cache/huggingface/hub/models--Qwen--Qwen2-Audio-7B-Instruct/snapshots/0a095220c30b7b31434169c3086508ef3ea5bf0a"

def qwen2audio_timbre_range_inference_ft():
    output_path = script_dir.parent / ".." / ".." / "data" / f"finetune_qwen2audio_timbre_range_post.csv"
    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
        audio_index, predicted_ranges = list(df["audio_index"]), list(df["timbre_range_post_ft"])
    else:
        df=pd.DataFrame(columns=["audio_index", "timbre_range_post_ft"])
        df.to_csv(output_path, index=False)
        audio_index, predicted_ranges = [], []

    engine = PtEngine(
    model_id_or_path=original_model_path,
    adapters=[adapter_path],
    device_map="cpu"
    )
    
    initial_idx = len(predicted_ranges)
    for idx in tqdm(range(initial_idx, 200)):
        infer_request = InferRequest(messages=[
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": f"/Users/jose_cruzado/Library/CloudStorage/OneDrive-TheUniversityofChicago/UChicago/MS Statistics/Courses/Spring 2025/Speech Technologies/Project/qwen2-audio-eval/data/finetunning_data/data/audio_clips/train_{str(idx).zfill(4)}.wav"},
                    {"type": "text", "text": "Listen the audio and classify the timbre range: answer 0 for narrow, 1 for moderate and 2 for wide. Answer only with a number"}
                ]
            }
        ])
        request_config = RequestConfig(max_tokens=16, temperature=0.0)
        response = engine.infer([infer_request], request_config)

        audio_index.append(idx)
        predicted_ranges.append(response[0].choices[0].message.content)
        df=pd.DataFrame(columns=["audio_index", "timbre_range_post_ft"])
        df["audio_index"] = audio_index
        df["timbre_range_post_ft"] = predicted_ranges
        df.to_csv(output_path, index=False)

if __name__ == "__main__":
    qwen2audio_timbre_range_inference_ft()