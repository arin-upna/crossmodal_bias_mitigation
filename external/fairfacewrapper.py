import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np
from pathlib import Path

device = torch.device('cuda:2')

model_fair_7 = torchvision.models.resnet34(pretrained=False)
model_fair_7.fc = nn.Linear(model_fair_7.fc.in_features, 18)
model_fair_7.load_state_dict(torch.load(Path(__file__).parent / 'fairface' / 'fair_face_models' / 'fairface_alldata_20191111.pt'
, map_location=device))
model_fair_7 = model_fair_7.to(device)
model_fair_7.eval()

trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_fairface(image):
    image_tensor = trans(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model_fair_7(image_tensor)
        outputs = outputs.cpu().numpy()[0]
    
    race_logits = outputs[:7]
    gender_logits = outputs[7:9]
    age_logits = outputs[9:18]
    
    race_scores = np.exp(race_logits) / np.sum(np.exp(race_logits))
    gender_scores = np.exp(gender_logits) / np.sum(np.exp(gender_logits))
    age_scores = np.exp(age_logits) / np.sum(np.exp(age_logits))
    
    race_pred = np.argmax(race_scores)
    gender_pred = np.argmax(gender_scores)
    age_pred = np.argmax(age_scores)
    
    race_labels = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
    gender_labels = ['male', 'female']
    age_labels = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
    
    return {
        'race': race_labels[race_pred],
        'gender': gender_labels[gender_pred],
        'age': age_labels[age_pred],
        'race_logits': race_logits,
        'gender_logits': gender_logits,
        'age_logits': age_logits,
        'race_scores': race_scores,
        'gender_scores': gender_scores,
        'age_scores': age_scores
    }