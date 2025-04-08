import torch.nn as nn

def get_updated_network(old: nn.Module, new: nn.Module, lr: float, load: bool = False) -> nn.Module:
    """
    Apply one manual gradient-based parameter update to a model.
    Typically used in meta-learning inner loops.

    Args:
        old (nn.Module): The original model with gradients.
        new (nn.Module): The new model to receive updated parameters.
        lr (float): Inner-loop learning rate (alpha).
        load (bool): If True, load the updated state directly. Otherwise, assign recursively via put_theta.

    Returns:
        nn.Module: The updated model.
    """
    updated_theta = {}
    current_weights = old.state_dict()
    grad_params = dict(old.named_parameters())

    for key, value in current_weights.items():
        if key in grad_params and grad_params[key].grad is not None:
            updated_theta[key] = grad_params[key] - lr * grad_params[key].grad
        else:
            updated_theta[key] = value

    return new.load_state_dict(updated_theta) if load else put_theta(new, updated_theta)


def put_theta(model: nn.Module, theta: dict) -> nn.Module:
    """
    Recursively assign updated weights to a model.

    Args:
        model (nn.Module): Model to update.
        theta (dict): Dictionary of parameter names to new values.

    Returns:
        nn.Module: Updated model.
    """
    def recursive_assign(module: nn.Module, prefix: str = ""):
        for name, child in module._modules.items():
            new_prefix = f"{prefix}.{name}" if prefix else name
            recursive_assign(child, new_prefix)

        for name, param in module._parameters.items():
            if param is not None:
                key = f"{prefix}.{name}" if prefix else name
                if key in theta:
                    module._parameters[name] = theta[key]

    recursive_assign(model)
    return model
