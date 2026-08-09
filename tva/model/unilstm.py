import torch
import torch.nn as nn

__all__ = ['UniLSTM']


class UniLSTM(nn.Module):
    '''Unidirectional LSTM for causal CTC sequence modelling.

    Unlike a bidirectional LSTM, an output at time ``t`` only uses recurrent
    context from time ``t`` and earlier. This makes it useful for testing
    whether future context shifts CTC character emissions toward the start of
    a handwriting recording.

    Args:
        size_in: Number of input features per time step.
        num_cls: Number of output classes, including the CTC blank.
        hidden_size: Hidden state dimension. Defaults to 128.
        num_layers: Number of stacked LSTM layers. Defaults to 3.
        r_drop: Dropout probability between LSTM layers. Defaults to 0.2.
    '''

    def __init__(
        self,
        size_in: int,
        num_cls: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        r_drop: float = 0.2,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            size_in,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=r_drop,
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_size, num_cls)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Return one class-probability vector per input time step.'''
        x, _ = self.lstm(x)
        x = self.fc(x)
        return self.softmax(x)
