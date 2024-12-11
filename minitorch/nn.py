from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    # new dimensions
    new_height = height // kh
    new_width = width // kw
    # split both height and width dimensions into new dimensions
    tiled = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    # permute the dimensions to match the new dimensions
    tiled = tiled.permute(0, 1, 2, 4, 3, 5).contiguous()
    # flatten the last two dimensions
    tiled = tiled.view(batch, channel, new_height, new_width, kh * kw)
    return tiled, new_height, new_width


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D."""
    batch, channel, height, width = input.shape
    # Use the existing tile function to reshape the input
    tiled, new_height, new_width = tile(input, kernel)

    # Calculate the average over the last dimension (kernel_height * kernel_width)
    pooled = tiled.mean(4)
    output = pooled.view(batch, channel, new_height, new_width)
    return output


reduce = FastOps.reduce(operators.max, float("-inf"))


def argmax(input: Tensor, dim: Tensor) -> Tensor:
    """Compute argmax as a one-hot tensor along the specified dimension."""
    max = reduce(input, int(dim.item()))
    return input == max


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for max operation along a dimension."""
        dim_int = int(dim.item())
        ctx.save_for_backward(argmax(input, dim))
        return reduce(input, dim_int)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for max operation."""
        (input,) = ctx.saved_values

        # # Create mask for positions equal to maximum
        # mask = argmax(input, dim_int)

        # Distribute gradient to maximum positions
        # grad_output = mask * grad_output

        return grad_output * input, 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction along a dimension."""
    # return Max.apply(input, tensor(dim))
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute softmax along a dimension."""
    # Subtract max for numerical stability
    x = input - max(input, dim)
    exp_x = x.exp()

    # Compute sum along dimension and normalize
    exp_sums = exp_x.sum(dim)

    # Reshape sum to broadcast
    shape = list(exp_x.shape)
    shape[dim] = 1
    exp_sums = exp_sums.contiguous().view(*shape)

    return exp_x / exp_sums


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of softmax along a dimension."""
    # Subtract max for numerical stability (LogSumExp trick)
    x = input - max(input, dim)

    # Compute exp and sum
    exp_x = x.exp()
    exp_sum = exp_x.sum(dim)

    # Reshape sum for broadcasting
    shape = list(exp_x.shape)
    shape[dim] = 1
    exp_sum = exp_sum.contiguous().view(*shape)

    # Compute log(softmax) = x - max(x) - log(sum(exp(x - max(x))))
    return x - exp_sum.log()


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D."""
    batch, channel, height, width = input.shape

    # Use tile function to reshape input into pooling windows
    tiled, new_height, new_width = tile(input, kernel)

    # Calculate max over the last dimension (kernel_height * kernel_width)
    pool = max(tiled, 4)

    # Reshape to final output dimensions
    output = pool.view(batch, channel, new_height, new_width)
    return output


def dropout(input: Tensor, prob: float, ignore: bool = False) -> Tensor:
    """Apply dropout to the input tensor."""
    if ignore:
        return input
    if prob == 1.0:
        return input * 0.0

    # Generate mask and scale by 1/(1-rate) to maintain expected value
    rand_values = rand(input.shape, input.backend, requires_grad=False)
    mask = (rand_values > prob) / (1.0 - prob)
    return input * mask
