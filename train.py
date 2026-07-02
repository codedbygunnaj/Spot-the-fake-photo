import os
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix


class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        img = self.subset.dataset.loader(path)
        img = self.transform(img)
        return img, label


def main(data_dir="/colab", epochs=25, out_path="/colab"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    base_dataset = datasets.ImageFolder(data_dir)  
    print(f"Classes found: {base_dataset.classes}")

    n_total = len(base_dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_subset, val_subset = random_split(base_dataset, [n_train, n_val])

    train_ds = SubsetWithTransform(train_subset, train_transforms)
    val_ds = SubsetWithTransform(val_subset, val_transforms)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    
    for p in model.features[-3:].parameters():
        p.requires_grad = True
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {"params": model.classifier.parameters(), "lr": 1e-4},
        {"params": model.features[-3:].parameters(), "lr": 1e-5},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3)

    best_acc, best_state = 0.0, None
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds += predicted.cpu().tolist()
                all_labels += labels.cpu().tolist()

        val_acc = 100 * correct / total if total else 0
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Val Accuracy: {val_acc:.1f}%")
        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, out_path)
            print(f"--> New best model saved with accuracy: {best_acc:.1f}%")
            print("    Confusion matrix [rows=true, cols=pred] (0=real,1=screen):")
            print(" ", confusion_matrix(all_labels, all_preds).tolist())

    print(f"\nDone. Best val accuracy: {best_acc:.1f}%  saved to {out_path}")


if __name__ == "__main__":
    main()