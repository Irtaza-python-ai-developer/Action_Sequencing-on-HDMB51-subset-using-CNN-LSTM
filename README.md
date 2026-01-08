# CNN–LSTM Action Recognition using Gradio

This repository contains a complete **CNN + LSTM based human action recognition system**
with an interactive **Gradio frontend**.

The project demonstrates:
- Deep learning model integration
- Temporal modeling using LSTM
- Frontend–backend communication using Gradio
- End-to-end execution from image input to annotated output

---

## Project Description

Human actions are temporal in nature. This project uses:
- a **Convolutional Neural Network (CNN)** to extract spatial features from video frames
- a **Long Short-Term Memory (LSTM)** network to model temporal relationships
- a **Gradio-based UI** for user interaction

For demonstration purposes, the system accepts a **single image** and repeats it internally
to simulate a short frame sequence required by the LSTM.

---

## Supported Actions

The system is trained to recognize the following actions:

- walk
- wave
- clap
- jump
- punch

---

## Dataset

**Dataset Used:** HMDB-51 (frame-based subset)

### Dataset Structure (expected)

```
HDMB51_Final/
├── walk/
│   ├── seq_01/
│   │   ├── frame1.jpg
│   │   ├── frame2.jpg
│   │   └── ...
├── jump/
├── punch/
├── wave/
└── clap/
```

### Dataset Location (local)

```
E:/Study/PhD/Deep Learning/Project/Project_HDMB_1/data/HDMB51_Final
```

⚠️ **Dataset is NOT included in this repository**
- Due to size limitations
- Due to dataset licensing restrictions

Dataset source:  
https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/

---

## Model Architecture

### CNN (Feature Extraction)
- ResNet18 (ImageNet pretrained)
- Removes final classification layer
- Extracts spatial features from each frame

### LSTM (Temporal Modeling)
- Processes sequence of CNN features
- Captures motion and temporal patterns

### Fully Connected Layer
- Outputs final action class

---

## Training Details

- Framework: PyTorch
- Optimizer: Adam
- Loss Function: CrossEntropyLoss
- Sequence length: 15 frames
- Hardware: GPU (CUDA)

---

## Inference Logic

- User uploads **one image**
- Image is resized and normalized
- Image is repeated to form a 15-frame sequence
- CNN extracts features per frame
- LSTM predicts the action
- Output image is annotated with predicted label

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

### requirements.txt
```
torch
torchvision
gradio
opencv-python
Pillow
numpy
```

---

## Run the Application

From the project root directory:

```bash
python app.py
```

You will see:

```
Running on local URL: http://127.0.0.1:7860
```

Open the link in a browser, upload an image, and view the result.

---

## Output

- Annotated image displaying predicted action
- Text output showing predicted class

⚠️ Accuracy may vary due to:
- Small dataset size
- Single-image inference for a temporal task

This is **expected behavior**.

---

## Repository Structure

```
.
├── app.py                  # Gradio frontend
├── src/
│   ├── dataset.py          # Frame dataset loader
│   ├── train.py            # Training script
│   └── inference.py        # Model inference + annotation
├── models/
│   ├── cnn_lstm.py         # CNN-LSTM architecture
│   ├── model.pth           # Trained weights
│   └── labels.json         # Label mapping
├── samples/                # Sample images
├── requirements.txt
└── README.md
```

---

## Notes for Evaluation / Viva

- CNN handles spatial features
- LSTM handles temporal dependencies
- Gradio enables easy frontend interaction
- Dataset excluded due to size and licensing
- Pipeline correctness is the focus, not accuracy

---

## Acknowledgement

HMDB-51 Dataset  
Serre Lab, Brown University

---

## Author

Developed as an academic coursework project demonstrating
CNN-LSTM based action recognition with Gradio integration.
