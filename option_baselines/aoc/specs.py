import dataclasses

import torch


@dataclasses.dataclass
class OptionExecutionState:
    executing_option: torch.Tensor
    option_is_available: torch.Tensor
