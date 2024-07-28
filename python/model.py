import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

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
        hidden_sizes: list[int],
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

class FCNet(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        hidden_sizes: list[int],
        input_dropout: float,
        hidden_dropout: float,
        output_size: int = 1
    ) -> None:
        super().__init__()

        layers_ = [input_size] + hidden_sizes
        _layers = hidden_sizes + [output_size]

        self.num_params = np.sum(np.array(layers_) * np.array(_layers) + np.array(_layers))
        self.num_layers = len(hidden_sizes) + 1
        self.name = f'{self.num_layers}-Layer FC-Net ({self.num_params} parameters, dropout = {hidden_dropout})'

        layers = zip(
            [input_dropout] + [hidden_dropout for _ in hidden_sizes],
            layers_,
            _layers
        )

        for layer_num, (dropout, dim_in, dim_out) in enumerate(layers):
            setattr(
                self,
                f'dropout_{layer_num}',
                nn.Dropout(dropout)
            )
            setattr(
                self,
                f'fc_{layer_num}',
                nn.Linear(dim_in, dim_out)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        for layer_num in range(self.num_layers):
            dropout = getattr(
                self,
                f'dropout_{layer_num}',
            )
            fc = getattr(
                self,
                f'fc_{layer_num}'
            )
            x = dropout(x)
            x = fc(x)
            if layer_num + 1 < self.num_layers:
                x = F.relu(x)
        
        return x
