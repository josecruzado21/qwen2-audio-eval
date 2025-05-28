
import os
from torchaudio.datasets import LIBRISPEECH

root_path = os.path.expanduser("~/librispeech_dev_clean")
os.makedirs(root_path, exist_ok=True)

val_set = LIBRISPEECH(
    root=root_path,
    url="dev-clean",
    download=True
)