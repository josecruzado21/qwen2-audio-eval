import os
os.environ["TTS_HOME"] = os.path.expanduser("~/.cache")
from TTS.api import TTS

tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)