import torch.nn as nn

class CustomClass(nn.Module):
    def __init__(self, hidden_neurons,NumberOfOutputs):
        super().__init__()
        self.Sequence = nn.Sequential(
            nn.Linear(1,hidden_neurons),
            nn.CELU(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.CELU(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.LeakyReLU(),
            nn.Linear(hidden_neurons,hidden_neurons),
            nn.CELU(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.CELU(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.CELU(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.LeakyReLU(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.CELU(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.CELU(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.CELU(),
            nn.Linear(hidden_neurons,NumberOfOutputs)
        )
    def forward(self,X):
        return self.Sequence(X)