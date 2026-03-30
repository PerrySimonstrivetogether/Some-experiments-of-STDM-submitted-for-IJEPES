import copy
from typing import Dict, List, Tuple

import torch


def transfer_compatible_parameters(
    source_state: Dict[str, torch.Tensor],
    target_state: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], List[str], List[str]]:
    """
    Copy only shape-compatible tensors from a source model to a target model.

    This is the key step used in cross-network transfer when the source and
    target feeders do not share exactly the same state dimension.
    """
    new_target_state = copy.deepcopy(target_state)
    loaded_names: List[str] = []
    skipped_names: List[str] = []
    for name, tensor in source_state.items():
        if name in new_target_state and new_target_state[name].shape == tensor.shape:
            new_target_state[name] = tensor.detach().clone()
            loaded_names.append(name)
        else:
            skipped_names.append(name)
    return new_target_state, loaded_names, skipped_names


def freeze_parameter_subset(model: torch.nn.Module, parameter_names: List[str]) -> None:
    parameter_set = set(parameter_names)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name not in parameter_set


def unfreeze_all(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True
