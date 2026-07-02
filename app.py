import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import time
import os

# Page Setup
st.set_page_config(page_title="Spot the Fake Photo", layout="centered")
st.title("📸 Live Recapture Detector")
st.write("Aim your camera at a real object or a digital screen.")

# Load Model (Cached so it doesn't reload every time you take a photo)
@st.cache_resource
def load_model():
    device = torch.device("cpu") # Web apps run best on CPU
    
    # 1. Rebuild the exact same structure used in your new train.py
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    
    # 2. Load the newly trained weights (Notice the updated filename)
    model_path = "mobile_net_best.pth"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find '{model_path}'. Make sure it is downloaded in this folder!")
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"🚨 Model loading error: {e}")
    st.stop()

# Exact same deterministic transforms used for your validation set in the new train.py
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# The Magic Widget: Browser Webcam!
img_file_buffer = st.camera_input("Take a photo")

if img_file_buffer is not None:
    # 1. Read Image
    img = Image.open(img_file_buffer).convert('RGB')
    
    # 2. Predict & Measure Latency
    start_time = time.time()
    input_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        screen_prob = probabilities[1].item() # Index 1 is 'screen'
        
    latency_ms = (time.time() - start_time) * 1000
    
    # 3. Display Results
    st.markdown("### Result:")
    if screen_prob >= 0.5:
        st.error(f"🚨 **FAKE! (Photo of a Screen)** \n\nConfidence: {screen_prob*100:.1f}%")
    else:
        st.success(f"✅ **REAL PHOTO!** \n\nConfidence: {(1-screen_prob)*100:.1f}%")
        
    st.caption(f"⚡ Latency: {latency_ms:.2f} ms")