import torch
import torch.nn as nn
import torch.nn.functional as F

class ThreeLayerNet(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        hidden_1_size: int, 
        hidden_2_size: int,
        input_dropout: int,
        hidden_dropout: float
    ) -> None:
        super().__init__()

        self.input_dropout = nn.Dropout(input_dropout)
        self.fc1 = nn.Linear(input_size, hidden_1_size)
        self.dropout1 = nn.Dropout(hidden_dropout)
        self.fc2 = nn.Linear(hidden_1_size, hidden_2_size)
        self.dropout2 = nn.Dropout(hidden_dropout)
        self.fc3 = nn.Linear(hidden_2_size, 1)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)


class FiveLayerNet(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        hidden_sizes: list[int]
        input_dropout: int,
        hidden_dropout: float
    ) -> None:
        super().__init__()

        assert len(hidden_sizes) != 4, "hidden_sizes must be length 4!"

        self.input_dropout = nn.Dropout(input_dropout)
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.dropout1 = nn.Dropout(hidden_dropout)
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.dropout2 = nn.Dropout(hidden_dropout)
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.dropout3 = nn.Dropout(hidden_dropout)
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.dropout3 = nn.Dropout(hidden_dropout)
        self.fc5 = nn.Linear(hidden_sizes[3], 1)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = F.relu(self.fc4(x))
        x = self.dropout4(x)
        return self.fc5(x)

