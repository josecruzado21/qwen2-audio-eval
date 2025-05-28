import subprocess
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_ENABLE_MPS"] = "0"
os.environ["DEVICE"] = "cpu"
import torch
torch.device("cpu")
from accelerate import Accelerator
acc = Accelerator(cpu=True)

current_file_path = os.path.abspath(__file__)
root_path = os.path.abspath(os.path.join(current_file_path, '../../..'))
def run_finetunning():
    command = [
        "swift", "sft",
        "--model_type", "qwen2_audio",
        "--model", "/Users/jose_cruzado/.cache/huggingface/hub/models--Qwen--Qwen2-Audio-7B-Instruct/snapshots/0a095220c30b7b31434169c3086508ef3ea5bf0a",
        "--dataset", os.path.join(root_path, "data", "finetunning_data", "train.json"),
        "--output_dir", "./outputs/qwen2audio-ft",
        "--bf16", "False",
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "4",
        "--num_train_epochs", "3",
        "--save_steps", "1",
        "--eval_steps", "1",
        "--logging_steps", "1",
        "--device_map", "cpu",
        "--dataloader_num_workers", "0",
        "--torch_dtype", "float32",
        "--do-train",
    ]

    subprocess.run(command)

if __name__ == "__main__":
    run_finetunning()