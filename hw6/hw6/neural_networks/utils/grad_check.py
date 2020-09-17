import numpy as np
from numpy.linalg import norm
from typing import Callable


def check_gradients(
    fn: Callable[[np.ndarray], np.ndarray],
    grad: np.ndarray,
    x: np.ndarray,
    dLdf: np.ndarray,
    h: float = 1e-6,
) -> float:
    """Performs numerical gradient checking by numerically approximating
    the gradient using a two-sided finite difference.

    For each position in `x`, this function computes the numerical gradient as:
        numgrad = fn(x + h) - fn(x - h)
                  ---------------------
                            2h

    Next, we use the chain rule to compute the derivative of the input of `fn`
    with respect to the loss:
        numgrad = numgrad @ dLdf

    The function then returns the relative difference between the gradients:
        ||numgrad - grad||/||numgrad + grad||

    Parameters
    ----------
    fn       function whose gradients are being computed
    grad     supposed to be the gradient of `fn` at `x`
    x        point around which we want to calculate gradients
    dLdf     derivative of
    h        a small number (used as described above)

    Returns
    -------
    relative difference between the numerical and analytical gradients
    """
    # ONLY WORKS WITH FLOAT VECTORS
    if x.dtype != np.float32 and x.dtype != np.float64:
        raise TypeError(f"`x` must be a float vector but was {x.dtype}")

    # initialize the numerical gradient variable
    numgrad = np.zeros_like(x)

    # compute the numerical gradient for each position in `x`
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h
        pos = fn(x).copy()
        x[ix] = oldval - h
        neg = fn(x).copy()
        x[ix] = oldval

        # compute the derivative, also apply the chain rule
        numgrad[ix] = np.sum((pos - neg) * dLdf) / (2 * h)
        it.iternext()

    return norm(numgrad - grad) / norm(numgrad + grad)
