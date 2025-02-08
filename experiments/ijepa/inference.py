# code to try the inference of thsi ijepa model

from PIL import Image
import torch
from torch.nn.functional import cosine_similarity
from transformers import AutoModel, AutoProcessor

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

path = '/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/dataset/first_stage_LLaVA/00261/002614054.jpg'
image = Image.open(path).convert('RGB')

model_id = "/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/hf_models/facebook/ijepa_vith14_22k/models--facebook--ijepa_vith14_22k/snapshots/ba3c4513ca2b0f0c010f80ae2265b3dbe1083039"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)
model = model.to(device)

with torch.no_grad():  
    inputs = processor(image, return_tensors="pt").to(device)
    outputs = model(**inputs)
a = outputs.last_hidden_state.mean(dim=1) # [bsz, 1280]
b = outputs.last_hidden_state[:, 0] # [bsz, 1280]
outputs
