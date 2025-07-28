import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from Mnist import Classifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_dataset = DataLoader(test_dataset,batch_size = 64,shuffle =  False)

model = Classifier(28).to(device)
model.load_state_dict(torch.load("MNIST.pth"))

model.eval()
criterion = nn.CrossEntropyLoss()

test_loss =0
for image,label in tqdm(test_dataset):
    image,label = image.to(device),label.to(device)

    model_label = model(image)
    loss = criterion(model_label,label)
    test_loss+=loss.item()

print(test_loss)
    
