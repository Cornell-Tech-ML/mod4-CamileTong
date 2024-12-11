"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable


#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Multiply two floats."""
    return x * y


def id(x: float) -> float:
    """Identity function."""
    return x


def add(x: float, y: float) -> float:
    """Add two floats."""
    return x + y


def neg(x: float) -> float:
    """Negate a float."""
    return x * -1.0


def lt(x: float, y: float) -> float:
    """Compare two floats."""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Compare two floats."""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Compare two floats."""
    if x > y:
        return x
    return y


def is_close(x: float, y: float) -> float:
    """Check if two floats are close within a tolerance of 1e-2."""
    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:
    """Calculate the sigmoid of a float."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Calculate the ReLU of a float."""
    if x > 0:
        return x
    return 0.0


def log(x: float) -> float:
    """Calculate the natural logarithm of a float."""
    return math.log(x)


def exp(x: float) -> float:
    """Calculate the exponential of a float."""
    return math.exp(x)


def inv(x: float) -> float:
    """Calculate the inverse of a float."""
    return 1.0 / x


def log_back(x: float, y: float) -> float:
    """Calculate the derivative of the log function."""
    return y / (x + 1e-6)


def inv_back(x: float, y: float) -> float:
    """Calculate the derivative of the inverse function."""
    return -(1.0 / x**2) * y


def relu_back(x: float, y: float) -> float:
    """Calculate the derivative of the ReLU function times y."""
    return y if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(f: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Apply a function to each element of an iterable."""

    def _map(xs: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in xs:
            ret.append(f(x))
        return ret

    return _map


def zipWith(
    f: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Apply a function to each element of two iterables."""

    def _zipWith(xs: Iterable[float], ys: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(xs, ys):
            ret.append(f(x, y))
        return ret

    return _zipWith


def reduce(
    f: Callable[[float, float], float], initial: float
) -> Callable[[Iterable[float]], float]:
    """Reduce an iterable to a single value."""

    def _reduce(ls: Iterable[float]) -> float:
        val = initial
        for x in ls:
            val = f(val, x)
        return val

    return _reduce


def negList(xs: Iterable[float]) -> Iterable[float]:
    """Negate a list."""
    return map(neg)(xs)


def addLists(xs: Iterable[float], ys: Iterable[float]) -> Iterable[float]:
    """Add two lists together."""
    return zipWith(add)(xs, ys)


def sum(xs: Iterable[float]) -> float:
    """Sum a list."""
    return reduce(add, 0.0)(xs)


def prod(xs: Iterable[float]) -> float:
    """Take the product of a list."""
    return reduce(mul, 1.0)(xs)
