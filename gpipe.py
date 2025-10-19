import torch 
from mpi4py import MPI 
from torchvision import datasets, transforms
import torch.nn as nn 
import torch.optim as optim 
from model import model, loss_fn
from torch.utils.data import DataLoader
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



class TrainingPipe:
    def __init__(self, model, size, rank):
        self.rank = rank 
        self.size = size 
        self.original_model = model
        
        all_layers = list(model.net.children())
        total_layers = len(all_layers)

        stages = []
        i = 0 
        while i < len(all_layers):
            stage = [all_layers[i]]

            if isinstance(all_layers[i], nn.Linear) and i+1 < len(all_layers):
                if isinstance(all_layers[i+1], nn.ReLU):
                    stage.append(all_layers[i+1])
                    i += 2 
                else:
                    i += 1
            else:
                i += 1 
            stages.append(stage)
        print(f"Rank {rank}: Total Stages: {len(stages)}")

        stages_per_rank = len(stages) // size 
        start = rank * stages_per_rank
        end = start + stages_per_rank if rank < size-1 else len(stages)
        
        my_layers_list = []
        for stage in stages[start:end]:
            my_layers_list.extend(stage)

        self.my_layers = nn.Sequential(*my_layers_list)
        
        self.is_first = (rank == 0)
        self.is_last = (rank == size - 1)
        self.saved_input = []
        self.saved_output = []


    def forward(self, x):
        if self.is_first:
            x = x.view(x.size(0), -1)

        if not self.is_first:
            shape = comm.recv(source=self.rank-1, tag=10)
            x_numpy = np.empty(shape, dtype=np.float32)
            comm.Recv(x_numpy, source=self.rank-1, tag=11)
            x = torch.from_numpy(x_numpy)
            
        x.requires_grad = True
        self.saved_input.append(x)
        output = self.my_layers(x)
        output.retain_grad() 
        self.saved_output.append(output)

        if not self.is_last:
            comm.send(output.shape, dest=self.rank+1, tag=10)
            output_numpy = output.detach().numpy()
            comm.Send(output_numpy, dest=self.rank+1, tag=11)
            return None 
        return output 

    def backward(self):
        if self.is_last:
            if not self.is_first:
                for input in self.saved_input:
                    grad_input = input.grad
                    if grad_input is not None:
                        print(f"{self.rank} sending gradient to {self.rank-1}")
                        comm.send(grad_input.shape, dest=self.rank-1, tag=20)
                        comm.Send(grad_input.detach().numpy(), dest=self.rank-1, tag=21)
        
        elif not self.is_first:
            print(f"{self.rank} Recieved gradient from {self.rank+1}")
            for i in range(len(self.saved_output)):
                shape = comm.recv(source=self.rank+1, tag=20)
                grad_numpy = np.empty(shape, dtype=np.float32)
                comm.Recv(grad_numpy, source=self.rank+1, tag=21)
                grad_output = torch.from_numpy(grad_numpy)
                
                self.saved_output[i].backward(grad_output, retain_graph=(i < len(self.saved_output)-1))

                if not self.is_first:
                    grad_input = self.saved_input[i].grad
                    if grad_input is not None:
                        print(f"Rank {self.rank}: sending gradient to rank {self.rank-1}")
                        comm.send(grad_input.shape, dest=self.rank-1, tag=20)
                        grad_numpy = grad_input.detach().numpy()
                        comm.Send(grad_numpy, dest=self.rank-1, tag=21)
        else:
            print(f"{self.rank}: Recieved Gradient from {self.rank+1}")
            for i in range(len(self.saved_output)):
                shape = comm.recv(source=self.rank+1, tag=20)
                grad_numpy = np.empty(shape, dtype=np.float32)
                comm.Recv(grad_numpy, source=self.rank+1, tag=21)
                grad_output = torch.from_numpy(grad_numpy)
                
                self.saved_output[i].backward(grad_output, retain_graph=(i < len(self.saved_output)-1))
            
    def zero_grad(self, optimizer):
        optimizer.zero_grad()
        self.saved_input = []
        self.saved_output = []

    def step(self, optimizer):
        optimizer.step()



num_min_batches = 8
new_model = TrainingPipe(model, size, rank)

optimizer = optim.AdamW(params=new_model.my_layers.parameters(), lr=1e-3)

if rank == 0:
    train_dataset, test_dataset = load_data(transform, download=False)
    comm.Barrier()
else:
    comm.Barrier()
    train_dataset, test_dataset = load_data(transform, download=False)


if rank == 0:
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    total_batches = len(train_dataloader)
    
    for r in range(1, size):
        comm.send(total_batches, dest=r, tag=80)
        

    for epoch in range(10):
        for batch_idx, (x, y) in enumerate(train_dataloader):
            mini_batch_x = torch.chunk(x, num_min_batches, dim=0)
            mini_batch_y = torch.chunk(y, num_min_batches, dim=0)

            for idx, (xb, yb) in enumerate(zip(mini_batch_x, mini_batch_y)):
                _ = new_model.forward(xb)
                comm.Send(yb.numpy(), dest=size-1, tag=99)
                
            new_model.backward()
            new_model.step(optimizer)
            new_model.zero_grad(optimizer)
            if (batch_idx%10==0):
                print(f"Rank: {rank}, Epoch: {epoch+1}, Batch: {batch_idx}")

elif rank == size-1:
    total_batches = comm.recv(source=0, tag=80)

    for epoch in range(10):
        for batch_idx in range(total_batches):
            for id in range(num_min_batches):
                output = new_model.forward(None)
                
                y_numpy = np.empty((output.size(0),), dtype=np.int64)
                comm.Recv(y_numpy, source=0, tag=99)
                y=torch.from_numpy(y_numpy)

                loss = loss_fn(output, y)
                loss.backward()
            
            new_model.backward()
            new_model.step(optimizer)
            new_model.zero_grad(optimizer)

            if (batch_idx % 10 == 0):
                print(f"Rank {rank}: Epoch: {epoch+1}, loss: {loss.item():.2f}")
    comm.Barrier()
                

else:
    total_batches = comm.recv(source=0, tag=80)

    for epoch in range(10):
        for batch_idx in range(total_batches):
            for idx in range(num_min_batches):
                _ = new_model.forward(None)

            new_model.backward()
            new_model.step(optimizer)
            new_model.zero_grad(optimizer)
            if (batch_idx%10 == 0):
                print(f"Rank: {rank}, Epoch: {epoch+1}, Batch: {batch_idx}")
    comm.Barrier()

