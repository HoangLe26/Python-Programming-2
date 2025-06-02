import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Thiết lập device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Chuẩn bị dữ liệu với Data Augmentation
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Lật ngang ngẫu nhiên
    transforms.RandomRotation(10),      # Xoay ngẫu nhiên trong khoảng [-10, 10] độ
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# Chia tập train thành train và validation
train_size = int(0.8 * len(trainset))
val_size = len(trainset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 2. Xây dựng mô hình MLP với Dropout
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Thêm Dropout với tỷ lệ 0.5

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 3. Hàm huấn luyện với Early Stopping
def train_model(model, trainloader, valloader, testloader, epochs=20, patience=3):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)  # Giảm learning rate và thêm Weight Decay
    
    train_losses, val_losses, test_losses = [], [], []
    train_accs, val_accs, test_accs = [], [], []
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Huấn luyện
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_losses.append(running_loss / len(trainloader))
        train_accs.append(100 * correct / total)
        
        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_losses.append(val_loss / len(valloader))
        val_accs.append(100 * correct / total)
        
        # Test
        test_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_losses.append(test_loss / len(testloader))
        test_accs.append(100 * correct / total)
        
        # In kết quả
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.2f}%, '
              f'Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accs[-1]:.2f}%, '
              f'Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accs[-1]:.2f}%')
        
        # Early Stopping
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1} due to no improvement in Val Loss.')
            model.load_state_dict(best_model_state)
            break
    
    return train_losses, val_losses, test_losses, train_accs, val_accs, test_accs

# 4. Hàm đánh giá và tính confusion matrix
def evaluate_model(model, dataloader, set_name="Test"):
    model = model.to(device)
    model.eval()
    y_true, y_pred = [], []
    loss, correct, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = loss / len(dataloader)
    accuracy = 100 * correct / total
    print(f"{set_name} Loss: {avg_loss:.4f}, {set_name} Accuracy: {accuracy:.2f}%")
    return y_true, y_pred, avg_loss, accuracy

# 5. Hàm vẽ learning curves
def plot_learning_curves(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs, title):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.plot(test_losses, label='Test Loss', color='green')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy', color='blue')
    plt.plot(val_accs, label='Validation Accuracy', color='orange')
    plt.plot(test_accs, label='Test Accuracy', color='green')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('mlp_learning_curves.png')
    plt.close()

# 6. Hàm vẽ confusion matrix
def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {title}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(filename)
    plt.close()

# 7. Chạy huấn luyện và đánh giá
mlp = MLP()
print("Training MLP...")
mlp_train_losses, mlp_val_losses, mlp_test_losses, mlp_train_accs, mlp_val_accs, mlp_test_accs = train_model(mlp, trainloader, valloader, testloader, epochs=20, patience=3)

# 8. Đánh giá và vẽ confusion matrix cho từng tập
print("\nEvaluating MLP on all sets...")
# Training set
mlp_train_y_true, mlp_train_y_pred, mlp_train_loss, mlp_train_acc = evaluate_model(mlp, trainloader, "Train")
plot_confusion_matrix(mlp_train_y_true, mlp_train_y_pred, 'Training', 'mlp_confusion_matrix_train.png')

# Validation set
mlp_val_y_true, mlp_val_y_pred, mlp_val_loss, mlp_val_acc = evaluate_model(mlp, valloader, "Validation")
plot_confusion_matrix(mlp_val_y_true, mlp_val_y_pred, 'Validation', 'mlp_confusion_matrix_val.png')

# Test set
mlp_test_y_true, mlp_test_y_pred, mlp_test_loss, mlp_test_acc = evaluate_model(mlp, testloader, "Test")
plot_confusion_matrix(mlp_test_y_true, mlp_test_y_pred, 'Test', 'mlp_confusion_matrix_test.png')

# In kết quả cuối cùng
print(f"\nFinal MLP Results:")
print(f"Train Loss: {mlp_train_loss:.4f}, Train Accuracy: {mlp_train_acc:.2f}%")
print(f"Validation Loss: {mlp_val_loss:.4f}, Validation Accuracy: {mlp_val_acc:.2f}%")
print(f"Test Loss: {mlp_test_loss:.4f}, Test Accuracy: {mlp_test_acc:.2f}%")

# 9. Vẽ biểu đồ learning curves
plot_learning_curves(mlp_train_losses, mlp_val_losses, mlp_test_losses, mlp_train_accs, mlp_val_accs, mlp_test_accs, 'MLP')