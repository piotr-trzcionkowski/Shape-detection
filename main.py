import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
import torch.nn.functional as F
import torchmetrics
import kagglehub
import cv2
import os

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
    h, w = roi.shape
    canvas = np.ones((size, size), dtype=np.uint8)  # białe tło
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = roi
    return canvas

class ShapeImage:
    def __init__(self, image_pil):
        self.image_pil = image_pil
        self.detected_shapes = []
        self.uncertain_shapes = []
        self.result_img = None

    def detect_and_classify_shapes(self, model, padding_ratio=0.2, confidence_threshold=0.6):
        image_np = np.array(self.image_pil.convert("RGB"))
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
            shape_info = {
                "bbox": (x_pad, y_pad, w_pad, h_pad),
                "class": pred.item(),
                "confidence": confidence.item(),
                "roi": roi_centered
            }
            if confidence.item() > confidence_threshold:
                self.detected_shapes.append(shape_info)
                cv2.rectangle(result_img, (x_pad, y_pad), (x_pad+w_pad, y_pad+h_pad), color, 2)
            else:
                self.uncertain_shapes.append(shape_info)
        self.result_img = result_img

    def get_uncertain_count(self):
        return len(self.uncertain_shapes)

    def get_detected_count(self):
        return len(self.detected_shapes)

    def get_result_image(self):
        return self.result_img

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
    model.load_state_dict(torch.load(model_path))
    model.eval()
    st.sidebar.success("Model został wczytany z pliku.")
else:
    st.sidebar.warning("Brak zapisanego modelu – trwa trening...")
    num_epochs = 20
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
    st.sidebar.success("Model został wytrenowany i zapisany.")

# Set up of multiclass accuracy metric
def compute_metrics(model, test_loader):
    acc = torchmetrics.Accuracy(task="multiclass", num_classes=3)
    precision = torchmetrics.Precision(task="multiclass", num_classes=3, average='macro')
    recall = torchmetrics.Recall(task="multiclass", num_classes=3, average='macro')
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            acc(preds, labels)
            precision(preds, labels)
            recall(preds, labels)
    return acc.compute(), precision.compute(), recall.compute()

# Wywołanie na starcie
if "test_accuracy" not in st.session_state:
    st.session_state.test_accuracy, st.session_state.test_precision, st.session_state.test_recall = compute_metrics(model, test_loader)
st.sidebar.write(f"Test accuracy: \n{st.session_state.test_accuracy}")
st.sidebar.write(f"Test precision: \n{st.session_state.test_precision}")
st.sidebar.write(f"Test recall: \n{st.session_state.test_recall}")


st.set_page_config(page_title="Wykrywanie ksztaltow", layout="wide")
st.title("Wykrywanie ksztaltow")

# ---------- Panel boczny ---------- #
st.sidebar.markdown("### Legenda kształtów")
st.sidebar.markdown("- 🟥 **Kwadrat** (czerwony obrys)")
st.sidebar.markdown("- 🟩 **Koło** (zielony obrys)")
st.sidebar.markdown("- 🟦 **Trójkąt** (niebieski obrys)")

uploaded_file = st.file_uploader("Wczytaj obraz dokumentu", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    shape_img = ShapeImage(image)
    shape_img.detect_and_classify_shapes(model)
    st.image(shape_img.get_result_image(), caption="Wykryte kształty", width='stretch')
    uncertain_count = shape_img.get_uncertain_count()
    st.info(f"W detekcji {uncertain_count} kształtów nie spełnia warunku pewności.")

    # Resetowanie sesji przy nowym obrazie
    if "last_file" not in st.session_state or st.session_state.last_file != uploaded_file:
        st.session_state.manual_labels = []
        st.session_state.uncertain_index = 0
        st.session_state.labeling_active = False
        st.session_state.retrain_question = False
        st.session_state.show_labeling_question = True
        st.session_state.last_file = uploaded_file

    # 1. Pytanie o ręczną klasyfikację niepewnych kształtów
    if uncertain_count > 0 and not st.session_state.labeling_active and not st.session_state.retrain_question:
        if "show_labeling_question" not in st.session_state:
            st.session_state.show_labeling_question = True
        if st.session_state.show_labeling_question:
            st.write("Czy chcesz ręcznie oznaczyć niepewne kształty?")
            colA, colB = st.columns(2)
            tak_clicked = colA.button("TAK", key="start_labeling")
            nie_clicked = colB.button("NIE", key="skip_labeling")
            if tak_clicked:
                st.session_state.labeling_active = True
                st.session_state.show_labeling_question = False  # Ukryj pytanie i przyciski
            elif nie_clicked:
                st.session_state.labeling_active = False
                st.session_state.retrain_question = False
                st.session_state.show_labeling_question = False  # Ukryj pytanie i przyciski
                st.stop()

    # 2. Ręczna klasyfikacja niepewnych kształtów
    if st.session_state.labeling_active and st.session_state.uncertain_index < uncertain_count:
        shape = shape_img.uncertain_shapes[st.session_state.uncertain_index]
        roi = shape["roi"]
        st.image(roi * 255, caption="Niepewny kształt (wycinek)", width=400)
        st.write("Wybierz klasę kształtu:")
        col1, col2, col3, col4 = st.columns(4)
        if col1.button("Kwadrat", key=f"square_{st.session_state.uncertain_index}"):
            shape["class"] = 0
            st.session_state.manual_labels.append(shape)
            st.session_state.uncertain_index += 1
        if col2.button("Koło", key=f"circle_{st.session_state.uncertain_index}"):
            shape["class"] = 1
            st.session_state.manual_labels.append(shape)
            st.session_state.uncertain_index += 1
        if col3.button("Trójkąt", key=f"triangle_{st.session_state.uncertain_index}"):
            shape["class"] = 2
            st.session_state.manual_labels.append(shape)
            st.session_state.uncertain_index += 1
        if col4.button("Żaden z nich", key=f"none_{st.session_state.uncertain_index}"):
            # Usuń ten bounding box z listy niepewnych (pomijamy go)
            st.session_state.uncertain_index += 1

    # 3. Po ręcznym oznaczeniu wszystkich niepewnych kształtów pojawia się pytanie o retrening
    if st.session_state.labeling_active and st.session_state.uncertain_index >= uncertain_count and not st.session_state.retrain_question:
        st.session_state.retrain_question = True

    if st.session_state.retrain_question:
        st.write("Czy chcesz dołączyć obraz do zestawu treningowego i przetrenować model?")
        colTAK, colNIE = st.columns(2)
        retrain_tak = colTAK.button("TAK", key="retrain_yes")
        retrain_nie = colNIE.button("NIE", key="retrain_no")
        if retrain_nie or retrain_tak:
            st.session_state.labeling_active = False
            st.session_state.retrain_question = False
            st.session_state.uncertain_index = 0

            if retrain_tak:
                # Dodaj ręcznie oznaczone bounding boxy do danych treningowych
                new_imgs = []
                new_labels = []
                for shape in st.session_state.manual_labels:
                    new_imgs.append(shape["roi"].reshape(1, 25, 25))
                    new_labels.append(shape["class"])
                if len(new_imgs) > 0:
                    # Połącz z istniejącym datasetem
                    new_imgs_np = np.array(new_imgs, dtype=np.float32)
                    new_labels_np = np.array(new_labels, dtype=np.int64)
                    # Rozszerz istniejący dataset
                    dataset.data = np.concatenate([dataset.data, new_imgs_np.reshape(len(new_imgs), -1)], axis=0)
                    dataset.labels = np.concatenate([dataset.labels, new_labels_np], axis=0)
                    # Stwórz nowe DataLoadery
                    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
                    test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
                    # Trening modelu na rozszerzonym zbiorze
                    st.write("Trwa retrening modelu na nowych danych...")
                    num_epochs = 5
                    for epoch in range(num_epochs):
                        for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
                            data = data.to("cpu")
                            targets = targets.to("cpu")
                            scores = model(data)
                            loss = loss_fn(scores, targets)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                    torch.save(model.state_dict(), model_path)
                    st.success("Model został przetrenowany i zapisany.")
                    st.session_state.test_accuracy, st.session_state.test_precision, st.session_state.test_recall = compute_metrics(model, test_loader)
                else:
                    st.info("Nie dodano żadnych nowych danych do treningu.")

            st.session_state.manual_labels = []
            st.stop()
