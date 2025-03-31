import torch


class Optimizer:
    """
    A utility wrapper around the Adam optimizer that integrates a learning rate scheduler
    based on exponential decay.

    This class simplifies the optimization process by coupling gradient update,
    scheduling, and zeroing of gradients into a clean interface.

    Components:
        - Adam optimizer with custom betas and epsilon.
        - Learning rate scheduler with exponential decay every `decay_step` epochs.

    Args:
        parameter (iterable): The model parameters to optimize (e.g., model.parameters()).
        lr (float): Initial learning rate.
    """
    def __init__(self, parameter, lr):
        """
        Initialize the optimizer and scheduler.

        Args:
            parameter (iterable): Model parameters to optimize.
            lr (float): Initial learning rate for the optimizer.
        """
        self.optim = torch.optim.Adam(parameter, lr=lr, betas=(0.9, 0.9), eps=1e-12)
        decay, decay_step = 0.75, 1000
        l = lambda epoch: decay ** (epoch // decay_step)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=l)

    def step(self):
        """
        Perform one optimization step:
            1. Update model parameters.
            2. Advance the learning rate scheduler.
            3. Reset gradients to zero.
        """
        self.optim.step()
        self.schedule()
        self.optim.zero_grad()

    def schedule(self):
        """
        Advance the learning rate scheduler by one step.
        """
        self.scheduler.step()

    def zero_grad(self):
        """
        Reset all gradients of the optimized parameters.
        """
        self.optim.zero_grad()

    @property
    def lr(self):
        """
        Return the current learning rate (as a list, one per parameter group).

        Returns:
            list[float]: Current learning rate(s).
        """
        return self.scheduler.get_last_lr()
