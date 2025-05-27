import subprocess
import os

current_file_path = os.path.abspath(__file__)
root_path = os.path.abspath(os.path.join(current_file_path, '../../..'))
def run_training():
    command = [
        "swift", "sft",
        "--model_type", "qwen2-audio-7b-instruct",
        "--dataset_dir", os.path.join(root_path, "data", "finetunning_data"),
        "--model_id_or_path", "Qwen/Qwen2-Audio-7B-Instruct",
        "--train_dataset", "train.json",
        "--output_dir", "./outputs/qwen2audio-ft",
        "--bf16", "True",
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "4",
        "--num_train_epochs", "3",
        "--save_steps", "1",
        "--eval_steps", "1",
        "--logging_steps", "1"
    ]

    subprocess.run(command)

if __name__ == "__main__":
    run_training()