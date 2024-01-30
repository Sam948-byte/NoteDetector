# NOTE do this on a COPY of the weights, NOT the original
import torch
from ultralytics import YOLO

model = YOLO("scripts/v13copy.pt")
remap = {0:"robot", 1:"note"}
for k,v in remap.items():
    model.model.names[k] = v

torch.save(model.ckpt, model.model.pt_path) # overwrites the existing file