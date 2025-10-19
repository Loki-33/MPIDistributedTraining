import torch 
import torch.nn as nn 


class NN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1],hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], output_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)



input_dim=32*32*3
output_dim=10
hidden_dims=[512, 256, 128]
model = NN(input_dim, hidden_dims, output_dim)
loss_fn = nn.CrossEntropyLoss()

