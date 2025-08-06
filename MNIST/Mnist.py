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

class Classifier(nn.Module):
    def __init__(self,inp_dim):
       super(Classifier,self).__init__()
       self.model = nn.Sequential(
           nn.Flatten(),
           nn.Linear(inp_dim*inp_dim,128),
           nn.ReLU(),
           nn.Linear(128,64),
           nn.ReLU(),
           nn.Linear(64,10),
          
       )

    def forward(self,x):
        return self.model(x)
    


    
transform = transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

train_dataset = DataLoader(train_dataset,batch_size = 32,shuffle = True)



model = Classifier(28).to(device)
model.train() # as we want it to train only for now

crtierion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(),lr=0.01)


epoch =5
for i in range(epoch):
    train_loss = 0
    
    for image,label in tqdm(train_dataset):
        image,label = image.to(device),label.to(device)
        model_label = model(image)
        loss = crtierion(model_label,label)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        train_loss+= loss.item()
    
    print(f"Epoch [{i+1}/{epoch}], Loss: {train_loss:.4f}")

torch.save(model,"MNIST.pth")


