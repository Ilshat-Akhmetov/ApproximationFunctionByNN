import torch.nn as nn

class CustomClass(nn.Module):
    def __init__(self, NumberOfInputs, hidden_neurons,NumberOfOutputs):
        super().__init__()
        self.Sequence = nn.Sequential(
            nn.BatchNorm1d(NumberOfInputs),
            nn.Linear(NumberOfInputs,hidden_neurons),
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