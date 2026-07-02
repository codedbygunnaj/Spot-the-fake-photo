import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import time

def load_model(model_path, device):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 2)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

def main():
    image_path = "/colab/test.jpg"
    model_path = "/colab/mobile_net_best.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = load_model(model_path, device)
    except FileNotFoundError:
        print("Error: Model not found. Run train.py first.")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Could not read image: {e}")
        return

    input_tensor = transform(img).unsqueeze(0).to(device)

    # --- START LATENCY TIMER ---
    start_time = time.time()

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        screen_prob = probabilities[1].item()

    end_time = time.time()
    # --- END LATENCY TIMER ---

    latency_ms = (end_time - start_time) * 1000

    print(f"-> {screen_prob:.2f}")

    print(f"\n--- DEBUG INFO ---")
    if screen_prob >= 0.5:
        print(f"Prediction: SCREEN ({screen_prob*100:.1f}% confidence)")
    else:
        print(f"Prediction: REAL ({(1-screen_prob)*100:.1f}% confidence)")
    print(f"Latency: {latency_ms:.2f} ms")

if __name__ == "__main__":
    main()