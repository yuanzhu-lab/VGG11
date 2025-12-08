# %%
# Set-up
import os

path = r"./"
if os.path.exists(path):
    print("ok")

# %%
# (Python 3.12.2)
# Reference: https://www.datascienceweekly.org/tutorials/pytorch-mnist-load-mnist-dataset-from-pytorch-torchvision
import torch
import torchvision
import torchvision.datasets as datasets

# Resize the images to 32x32
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),torchvision.transforms.ToTensor(),])

mnist_trainset = datasets.MNIST(root=os.path.join(path, "q2"),train=True,download=True,transform=transform)
mnist_testset = datasets.MNIST(root=os.path.join(path, "q2"),train=False,download=True,transform=transform)

# Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)

# Loader
train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=64, shuffle=False)

# %%
import torch.nn as nn
import torch.optim as optim
import json

# VGG11 class
class VGG11(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG11, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x

# Model
model = VGG11().to(device)

# Cross-entropy loss
criterion = nn.CrossEntropyLoss()

# Adam Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Evaluation
def evaluate(model, loader):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss_sum += criterion(outputs, labels).item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return loss_sum / len(loader), correct / total

train_losses = []
test_losses = []
train_accs = []
test_accs = []

# Save checkpoints
save_dir = os.path.join(path, "q2", "checkpoints")
os.makedirs(save_dir, exist_ok=True)

def save_epoch(epoch, model, train_loss, train_acc, test_loss, test_acc):
    torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch + 1}.pth"))

    metrics = {
        "train_loss": train_losses,
        "train_acc": train_accs,
        "test_loss": test_losses,
        "test_acc": test_accs
    }
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Saved epoch {epoch + 1} checkpoint.")

# %%
# Training
EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    # epoch training results
    train_losses.append(running_loss / len(train_loader))
    train_accs.append(correct / total)

    # test results
    t_loss, t_acc = evaluate(model, test_loader)
    test_losses.append(t_loss)
    test_accs.append(t_acc)

    print(f"Epoch {epoch+1}: "
          f"Train Loss={train_losses[-1]:.4f}, Train Acc={train_accs[-1]*100:.2f}% | "
          f"Test Loss={t_loss:.4f}, Test Acc={t_acc*100:.2f}%")
    
    save_epoch(epoch, model,train_losses[-1], train_accs[-1],test_losses[-1], test_accs[-1])
