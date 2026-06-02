import torch
from torch import optim

import PRSNet

def train(num_epochs, data_iter):
    model = PRSNet.PRSNet()
    optimizer = optim.Adam(model.parameters())
    model.train()
    for epoch in range(num_epochs):
        for x in data_iter:
            optimizer.zero_grad()
            y1,y2,y3 = model(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.mps.is_available():
        device = torch.device("mps")




