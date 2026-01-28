import torch
from torch.autograd import Function

class GradientReversalFn(Function):
    """
    Gradient Reversal Layer (GRL) implementation.
    Forward pass: Identity transformation (returns input as is).
    Backward pass: Multiplies the gradient by a negative scalar (-alpha).
    """
    @staticmethod
    def forward(ctx, x, alpha):
        # Store alpha for the backward pass
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse the gradient sign and scale by alpha
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(torch.nn.Module):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()

    def forward(self, x, alpha=1.0):
        return GradientReversalFn.apply(x, alpha)