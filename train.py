import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import HandwritingCNN
import os

# 設定
BATCH_SIZE = 64
EPOCHS = 5
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data"

# データ前処理
transform = transforms.Compose([
    transforms.RandomRotation(5),
    transforms.RandomAffine(0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: torch.rot90(x, 1, [1, 2])),  # 画像の回転
    transforms.Lambda(lambda x: torch.flip(x, [1]))          # 左右反転
])

train_dataset = datasets.EMNIST(
    root=DATA_DIR,
    split="balanced",
    train=True,
    transform=transform,
    download=True
)
test_dataset = datasets.EMNIST(
    root=DATA_DIR,
    split="balanced",
    train=False,
    transform=transform,
    download=True
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# モデルの準備
NUM_CLASSES = 47  # EMNIST Balanced は47クラス
model = HandwritingCNN(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 学習ループ
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")

# テスト精度の確認
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# モデルの保存
os.makedirs("./models", exist_ok=True)
torch.save(model.state_dict(), "./models/emnist_cnn.pt")
print("Saved model to ./models/emnist_cnn.pt")
