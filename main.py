import os.path

import torch
from torch import optim
from Reader import CombinedDataLoader
from PRSLoss import *

import PRSNet

voxels_path = "./voxels/"
num_epochs = 1
device_name = "cpu"


def train(num_epochs, data_iter, model, optimizer, loss_func):
    optimizer = optim.Adam(model.parameters())
    model.train()
    for epoch in range(num_epochs):
        for voxels, points in data_iter:
            optimizer.zero_grad()
            planes, quaternions = model(voxels)
            loss = loss_func(planes, quaternions, points)


if __name__ == "__main__":
    dataloader = CombinedDataLoader(
        intermediate_data_dir='./shapenet_intermediate',
        batch_size=32,
        num_workers=0,  # 预加载时设置为 0
        shuffle=True,
        device=device_name
    )

    model = PRSNet.PRSNet().to(device_name)
    LossFunc = SymmetryLoss().to(device_name)
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    train(num_epochs, dataloader, model, optimizer, LossFunc)
