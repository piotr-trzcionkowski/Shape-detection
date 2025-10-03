import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# !pip install torchvision
import torchvision

import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# !pip install torchmetrics
import torchmetrics

import kagglehub
import cv2 

# Download latest version
path = kagglehub.dataset_download("vijay20213/shape-detection-circle-square-triangle")

print("Path to dataset files:", path)

loaded_data = pd.read_csv(path + "\\Shapes Dataset.csv")

class ShapeCSVDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.iloc[:, :-1].values.astype(np.float32)  # pixels
        self.labels = dataframe.iloc[:, -1].values.astype(np.int64)   # class

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].reshape(1, 25, 25)
        label = self.labels[idx]
        return torch.tensor(img), torch.tensor(label)
    
def center_and_resize(roi, size=25):
    # roi: binarny obrazek kształtu
    h, w = roi.shape
    canvas = np.ones((size, size), dtype=np.uint8)  # białe tło
    # Oblicz przesunięcie, aby wyśrodkować
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    # Wklej roi na środek
    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = roi
    return canvas

def detect_and_classify_shapes(image_pil, model, padding_ratio=0.2):
    image_np = np.array(image_pil.convert("RGB"))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_img = image_np.copy()
    colors = [(255,0,0), (0,255,0), (0,0,255)]  # kwadrat, koło, trójkąt

    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # Oblicz padding jako 10% wymiarów kształtu
        pad_w = int(w * padding_ratio)
        pad_h = int(h * padding_ratio)
        x_pad = max(x - pad_w, 0)
        y_pad = max(y - pad_h, 0)
        w_pad = min(w + 2 * pad_w, gray.shape[1] - x_pad)
        h_pad = min(h + 2 * pad_h, gray.shape[0] - y_pad)
        roi = gray[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
        roi_resized = cv2.resize(roi, (25, 25))
        _, roi_bin = cv2.threshold(roi_resized, 128, 1, cv2.THRESH_BINARY)
        roi_bin = 1 - roi_bin
        roi_centered = center_and_resize(roi_bin, 25)
        roi_tensor = torch.tensor(roi_centered, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            output = model(roi_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, pred = torch.max(probs, dim=1)
        color = colors[pred]
        if confidence.item() > 0.6:
            cv2.rectangle(result_img, (x_pad, y_pad), (x_pad+w_pad, y_pad+h_pad), color, 2)

    return result_img


batch_size = 600
dataset = ShapeCSVDataset(loaded_data)
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 6 * 6, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

model_path = "model.pth"

model = CNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


if os.path.exists(model_path):
    # Wczytaj zapisany model
    model.load_state_dict(torch.load(model_path))
    model.eval()
    st.sidebar.success("Model został wczytany z pliku.")
else:

    num_epochs=20
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")

        for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
            data = data.to("cpu")
            targets = targets.to("cpu")
            scores = model(data)
            loss = loss_fn(scores, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), model_path)
    


# Set up of multiclass accuracy metric
acc = torchmetrics.Accuracy(task="multiclass",num_classes=3)
precision = torchmetrics.Precision(task="multiclass", num_classes=3, average='macro')
recall = torchmetrics.Recall(task="multiclass", num_classes=3, average='macro')

# Iterate over the dataset batches
model.eval()
with torch.no_grad():
   for images, labels in test_loader:
       outputs = model(images)
       _, preds = torch.max(outputs, 1)
       acc(preds, labels)
       precision(preds, labels)
       recall(preds, labels)

# Compute total test metrics
test_accuracy = acc.compute()
test_precision = precision.compute()
test_recall = recall.compute()
st.sidebar.write(f"Test accuracy: \n{test_accuracy}")
st.sidebar.write(f"Test precision: \n{test_precision}")
st.sidebar.write(f"Test recall: \n{test_recall}")
# ---------- Ustawienia strony ---------- #
st.set_page_config(page_title="Wykrywanie ksztaltow", layout="wide")
st.title("Wykrywanie ksztaltow")

# ---------- Panel boczny ---------- #
    
# Dodaj legendę do sidebaru
st.sidebar.markdown("### Legenda kształtów")
st.sidebar.markdown("- 🟥 **Kwadrat** (czerwony obrys)")
st.sidebar.markdown("- 🟩 **Koło** (zielony obrys)")
st.sidebar.markdown("- 🟦 **Trójkąt** (niebieski obrys)")


uploaded_file = st.file_uploader("Wczytaj obraz dokumentu", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # st.image(image, caption="Wczytany obraz", use_container_width=True)
    # Detekcja i klasyfikacja kształtów
    result_img = detect_and_classify_shapes(image, model)
    st.image(result_img, caption="Wykryte kształty", use_container_width=True)




import io
buf = io.BytesIO()
# fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
st.download_button(
    "Pobierz obraz PNG",
    data=buf.getvalue(),
    # file_name=f"{spiral_type.lower()}_N{N}.png",
    mime="image/png",
)
