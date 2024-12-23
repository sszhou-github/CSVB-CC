import numpy as np
import torch
import clip
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load("ViT-B/32", device = device)

folder_path = 'datasets/RSOD/overpass/'

for filename in os.listdir(folder_path):

    image = preprocess(Image.open('datasets/RSOD/overpass/{}'.format(filename))).unsqueeze(0).to(device)
    
    #text = clip.tokenize(['aircraft', 'playground', 'oiltank', 'overpass']).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        torch.save(image_features,'data/overpass/{}'.format(filename.replace(".jpg", ".pt")))
        #text_features = model.encode_text(text)

        #logits_per_image, logits_per_text = model(image, text)
        #probs = logits_per_image.softmax(dim = 1).cpu().numpy()
