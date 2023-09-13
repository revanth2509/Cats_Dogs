import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from pathlib import Path
import os
import argparse
from model import CNNModel
from datasets import CustomeTransform

argument = argparse.ArgumentParser(description="Train data from path")
argument.add_argument("path",type=Path,help="A path to training dataset")
argument.add_argument("-t",type=str,help="train or test", default="train")
argument.add_argument("-e",type=int,help="epochs", default=10)
argument.add_argument("-b", type=int,help="batch size",default=128)
p = argument.parse_args()

path = p.path

transform = CustomeTransform(batch_size=128)
train_data = transform.loader(path=path)


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






