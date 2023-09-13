import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from pathlib import Path
import os
import argparse
from model import CNNModel

argument = argparse.ArgumentParser(description="Train data from path")
argument.add_argument("path",type=Path,help="A path to training dataset")
argument.add_argument("-t",type=str,help="train or test", default="train")
argument.add_argument("-e",type=int,help="epochs", default=10)
argument.add_argument("-b", type=int,help="batch size",default=128)
p = argument.parse_args()

path = p.path

dogs = list(Path(path).glob("dogs/*"))
cats = list(Path(path).glob("cats/*"))


transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(10),
    torchvision.transforms.Resize(500),
    torchvision.transforms.CenterCrop(500),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                          [0.229, 0.224, 0.225])
])

batch_size=p.b

model_data = torchvision.datasets.ImageFolder(path,transform=transform)
train_data = DataLoader(model_data,batch_size,shuffle=True)


model = CNNModel(3)
optim = torch.optim.Adam(params=model.parameters(),lr=0.0001)
loss = nn.CrossEntropyLoss()

# hist,loss1 = training_loop(20,model=model,optimizer=optim,loss_fn=loss,train_loader=train_dataset_loader,valid_dl=valid_dataset_loader)
device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)
# loss.to(device)
history = []
loss_val = []
for epoch in range(20):
    loss_train = 0
    for img, label in train_data:
        img = img.to(device)
        label = label.to(device)
        output = model(img)
        # print(output.squeeze(-1).shape,output.squeeze(-1))
        # print(output.shape)
        # print(label)
        # print(label.shape)
        loss1 = loss(output,label)
        # print(loss1)
        loss1.backward()
        optim.step()
        optim.zero_grad()
        
        loss_train+=loss1.item()
    loss_val.append(loss1)

    # if epoch == 1 or epoch % 10 == 0:
    # val = validation_step(valid_dataset_loader, model, loss)
    # print('{} Epoch {}, Training loss {}'.format(
    #     datetime.datetime.now(), epoch,
    #     loss_train / len(train_loader)))
    print(f"Epoch [{epoch}/{20}] => loss: {loss1}")






