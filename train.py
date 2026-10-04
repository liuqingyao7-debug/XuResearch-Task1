import os, json, time
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# basic settings
DATA_ROOT = os.path.expanduser("~/datasets/Dataset_BUSI_with_GT_Clean")
IMG_DIR = os.path.join(DATA_ROOT, "images")
OUT_DIR = os.path.expanduser("~/projects/research_training/outputs_5ep")
os.makedirs(OUT_DIR, exist_ok=True)

EPOCHS = 5
BATCH_SIZE = 16
LR = 1e-4

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


#data set
train_df = pd.read_excel(os.path.join(DATA_ROOT, "train.xlsx"))
test_df = pd.read_excel(os.path.join(DATA_ROOT, "test.xlsx"))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

class BUSIDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.img_dir, row["Image"])
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(row["Label"])
        return image, label

#data loader
train_set = BUSIDataset(train_df, IMG_DIR, transform)
test_set = BUSIDataset(test_df, IMG_DIR, transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


#model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)


#Loss functio, optimizer, training and evaluation functions
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

def train_one_epoch(model, loader):
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return total_loss / len(loader.dataset), all_preds, all_labels


#training loop 
history = {"train_loss": [], "test_loss": [], "test_acc": []}

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    train_loss = train_one_epoch(model, train_loader)
    test_loss, preds, labels = evaluate(model, test_loader)
    acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)

    history["train_loss"].append(train_loss)
    history["test_loss"].append(test_loss)
    history["test_acc"].append(acc)

    print(f"Epoch {epoch:02d}/{EPOCHS} | train loss {train_loss:.4f} | "
          f"test loss {test_loss:.4f} | test acc {acc:.4f} | {time.time() - t0:.1f}s",
          flush=True)

with open(os.path.join(OUT_DIR, "history.json"), "w") as f:
    json.dump(history, f)

torch.save(model.state_dict(), os.path.join(OUT_DIR, "resnet50_busi.pth"))

pd.DataFrame({"Image": test_df["Image"], "True": labels, "Pred": preds}) \
  .to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)

print("Done. Results saved to", OUT_DIR)