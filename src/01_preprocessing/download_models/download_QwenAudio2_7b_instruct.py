from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", 
                                          cache_dir = "/share/data/lang/users/ttic_31110/jcruzado/models/")
model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto", 
                                                           cache_dir = "/share/data/lang/users/ttic_31110/jcruzado/models/")