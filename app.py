import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# 1. Redéfinition de l'architecture (doit être identique au notebook)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.classifier(x)
        return x

# 2. Chargement du modèle
@st.cache_resource # Pour éviter de recharger le modèle à chaque clic
def load_model():
    model = SimpleCNN()
    # On charge sur le CPU pour que l'app soit plus universelle
    model.load_state_dict(torch.load('bone_fracture_cnn.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()
class_names = ['Fractured', 'Not Fractured']

# 3. Interface Streamlit
st.title("🦴 Détection de Fracture Osseuse")
st.write("Uploadez une image de rayon X pour analyse.")

uploaded_file = st.file_uploader("Choisir une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Image uploadée', use_container_width=True)
    
    # Prétraitement
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0)

    # Prédiction
    with st.spinner('Analyse en cours...'):
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            prediction = class_names[predicted.item()]

    # Affichage du résultat avec couleur
    if prediction == 'Fractured':
        st.error(f"Résultat : **{prediction}**")
    else:
        st.success(f"Résultat : **{prediction}**")