import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from torchvision.models import ResNet50_Weights

st.set_page_config(page_title="Reconhecimento de Digitais", layout="centered")

# Função para pré-processar imagens de digitais
def preprocess_fingerprint(image):
    image = image.convert("L")  # Converte para tons de cinza
    image = image.resize((224, 224))
    image_np = np.array(image)
    image_eq = cv2.equalizeHist(image_np)
    image_rgb = cv2.cvtColor(image_eq, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(image_rgb)

# Classe para extrair características da camada intermediária (layer4) da ResNet50
class FeatureExtractor(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.features = torch.nn.Sequential(*list(backbone.children())[:-2])  # Até layer4
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))  # Pool para reduzir a saída
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)

# Carrega o modelo e faz cache para reuso
@st.cache_resource
def load_model():
    resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model = FeatureExtractor(resnet)
    model.eval()
    return model

# Extrai o embedding da imagem
def extract_embedding(image, model):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(tensor).numpy()
    return embedding

# Calcula a similaridade entre dois vetores
def calculate_similarity(embedding1, embedding2):
    sim = cosine_similarity(embedding1, embedding2)
    return sim[0][0] * 100  # porcentagem

# Interface Streamlit
st.title("🔍 Reconhecimento de Digitais com IA")
st.write("Compare duas imagens de digitais e veja o nível de similaridade.")

# Upload de imagens
col1, col2 = st.columns(2)
with col1:
    img1 = st.file_uploader("Imagem 1", type=["jpg", "jpeg", "png", "tif", "tiff"], key="img1")
with col2:
    img2 = st.file_uploader("Imagem 2", type=["jpg", "jpeg", "png", "tif", "tiff"], key="img2")

# Quando ambas forem enviadas
if img1 and img2:
    image1 = preprocess_fingerprint(Image.open(img1))
    image2 = preprocess_fingerprint(Image.open(img2))

    st.image([image1, image2], caption=["Imagem 1", "Imagem 2"], width=250)

    model = load_model()
    emb1 = extract_embedding(image1, model)
    emb2 = extract_embedding(image2, model)

    similarity = calculate_similarity(emb1, emb2)

    st.markdown("### 🔎 Similaridade entre digitais:")
    st.metric(label="Resultado", value=f"{similarity:.2f}%")

    # Ajuste mais rigoroso nos thresholds
    if similarity > 88:
        st.success("As digitais são muito semelhantes!")
    elif similarity > 80:
        st.warning("As digitais têm alguma semelhança, mas podem não ser da mesma pessoa.")
    else:
        st.error("As digitais parecem diferentes.")
