from dataloader import MyDataset
import os
import torch
from torch.utils.data import Dataset, DataLoader
from Net import UNet
from torchvision import transforms
import torch.optim as optim
from train_test import train_cnnmodel, test_cnnmodel


path = os.getcwd()
data = MyDataset(path, transform=None)

Batch_Size = 4
Device = torch.device("cuda")
Epoch = 1000

train_size = int(len(data) * 0.7)
test_size = len(data) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])

pipeline = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = (0.1307,),std = (0.3081,))])

train_loader = DataLoader(train_dataset,batch_size=Batch_Size,shuffle=True,drop_last=False,num_workers=0,)
test_loader = DataLoader(test_dataset,batch_size=Batch_Size,shuffle=False,drop_last=False,num_workers=0)

model = UNet().to(Device)
optimizer = optim.Adam(model.parameters())

for epoch in range(1, Epoch+1):
    train_cnnmodel(model, Device, train_loader, optimizer, epoch)
    test_cnnmodel(model, Device, test_loader, epoch)

torch.save(model.state_dict(),'model.ckpt')



