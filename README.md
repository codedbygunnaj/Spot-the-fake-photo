# 📸 Spot the Fake Photo (Screen Recapture Detection)

## Overview
This project solves the "Screen Recapture" fraud problem. When users attempt to cheat a system by taking a photo of a digital screen (laptop/phone) instead of the real physical object, this lightweight model detects the subtle differences (Moiré patterns, screen glare, pixel grids) and flags it as a fake.

**🎯 Live Web Demo:** [Insert your Streamlit Cloud Link Here]

## Technical Approach
* **Algorithm:** Deep Learning / Transfer Learning
* **Model:** MobileNetV2 (Pre-trained on ImageNet, fine-tuned classification head & final block).
* **Why MobileNet?** It is designed specifically for mobile edge devices. It provides an excellent trade-off between high accuracy and ultra-low latency, running inference in just a few milliseconds even on a standard CPU.
* **Classes:** 
  * `0` = Real Photo
  * `1` = Screen Recapture

## Performance Metrics
* **Accuracy:** ~87-90% (Trained on a highly augmented, custom 200-image dataset to prevent overfitting).
* **Latency:** < 50ms per image on a standard CPU (Instantaneous for mobile deployment).
* **Cost at Scale:** $0 if deployed on-device. Extremely cheap if deployed on cloud due to the model's tiny memory footprint (~14MB).

## How to Run Locally

1. Clone this repository:
   ```bash
   git clone [https://github.com/codedbygunnaj/Spot-the-fake-photo.git]
   cd Spot-the-fake-photo
2. Install dependencies:
   pip install -r requirements.txt
3. Run the live camera demo:
   streamlit run app.py

## Future Improvements (With More Time & Data)
- Frequency Analysis Pipeline: Combine this CNN with a Fast Fourier Transform (FFT) preprocessing step to mathematically highlight Moiré patterns before feeding it to the neural network.
- Larger Dataset: Train on 10,000+ images across varied lighting conditions, screen types (OLED vs LCD), and focal lengths to push accuracy > 98%.
- Model Quantization: Convert the .pth weights to TFLite or ONNX and apply INT8 quantization to further reduce the model size to under 5MB for native Android/iOS deployment.
