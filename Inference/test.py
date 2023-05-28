import torch
from transformers import pipeline
import shutup
shutup.please()

pipe = pipeline(model="facebook/opt-1.3b", device_map="auto", model_kwargs={"load_in_8bit": True})
output = pipe("I am a humanoid robot called CURI", do_sample=True, top_p=0.95)
print(output)