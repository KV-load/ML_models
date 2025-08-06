import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.ToTensor()

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_dataset = DataLoader(test_dataset,batch_size = 64,shuffle =  False)

model =torch.load("MNIST.pth")

model.eval()
criterion = nn.CrossEntropyLoss()


correct = 0
total = 0

with torch.no_grad():
    for image, label in tqdm(test_dataset):
        image, label = image.to(device), label.to(device)
        outputs = model(image)
        loss = criterion(outputs, label)
        test_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

accuracy = 100 * correct / total
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {accuracy:.2f}%")
    
