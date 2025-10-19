import torch 
from mpi4py import MPI 
from torchvision import datasets, transforms
import torch.nn as nn 
import torch.optim as optim 
from model import model, loss_fn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler 
import numpy as np 


###-----------------------------------------------------------###############
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def load_data(transform, download=True):
    train_dataset = datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=download, 
        transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=download, 
        transform=transform
    )

    return train_dataset, test_dataset

def get_dataloader():
    if rank == 0:
        train_dataset, test_dataset = load_data(transform, download=True)
    comm.Barrier()
    if rank != 0:
        train_dataset, test_dataset = load_data(transform, download=False)


    train_sampler = DistributedSampler(train_dataset, num_replicas=size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=size, rank=rank, shuffle=False)

    train_dataloader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=32, sampler=test_sampler)

    return train_dataloader, test_dataloader

train_dataloader, test_dataloader = get_dataloader()
optimizer = optim.AdamW(params=model.parameters(), lr=1e-3)
for epoch in range(10):
    for x, y in train_dataloader:
        output = model(x)
        loss = loss_fn(output, y)
        optimizer.zero_grad()
        loss.backward()
        for param in model.parameters():
            if param.grad is not None:
                grad = param.grad.data 
                avg = comm.allreduce(grad, op=MPI.SUM)/size 
                param.grad.data=avg 
        optimizer.step()
    if rank==0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.2f}")
    comm.Barrier()

