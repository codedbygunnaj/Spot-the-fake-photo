import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import time
import os
from save_img import save

st.set_page_config(page_title="Spot the Fake Photo", layout="centered")

st.title("Live Recapture Detector")
st.write("Aim your camera at a real object or a digital screen.")

# Consent (default = False)
allow_collection = st.checkbox(
    "I consent to this image being stored anonymously for improving the model."
)

@st.cache_resource
def load_model():
    device = torch.device("cpu")

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 2)

    model_path = "mobile_net_best.pth"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find '{model_path}'.")

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


try:
    model = load_model()
except Exception as e:
    st.error(f" Model loading error: {e}")
    st.stop()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

img_file_buffer = st.camera_input("Take a photo")

if img_file_buffer is not None:

    img = Image.open(img_file_buffer).convert("RGB")

    start_time = time.time()

    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        screen_prob = probabilities[1].item()

    latency_ms = (time.time() - start_time) * 1000

    is_fake = screen_prob >= 0.5

    if allow_collection:
        save(img, is_fake)

    st.markdown("### Result:")

    if is_fake:
        st.error(f"**FAKE! (Photo of a Screen)**\n\nConfidence: {screen_prob*100:.1f}%")
    else:
        st.success(f"**REAL PHOTO!**\n\nConfidence: {(1-screen_prob)*100:.1f}%")

    st.caption(f"Latency: {latency_ms:.2f} ms")