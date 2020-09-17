import numpy as np


def im2col_indices(X, kernel_shape, stride, pad):
    p = pad

    n_examples, in_channels, in_rows, in_cols = X.shape
    kernel_height, kernel_width = kernel_shape

    out_rows = (in_rows + p[0] + p[1] - kernel_height) // stride + 1
    out_cols = (in_cols + p[2] + p[3] - kernel_width) // stride + 1

    i0 = np.repeat(np.arange(kernel_height), kernel_width)
    i0 = np.tile(i0, in_channels)
    i1 = stride * np.repeat(np.arange(out_rows), out_cols)
    j0 = np.tile(np.arange(kernel_width), kernel_height * in_channels)
    j1 = stride * np.tile(np.arange(out_cols), out_rows)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(in_channels), kernel_height * kernel_width).reshape(
        -1, 1
    )

    return i, j, k


def im2col(X, kernel_shape, stride, pad):
    n_channels = X.shape[3]
    kernel_height, kernel_width = kernel_shape
    X_pad, p = pad2d(X, pad, kernel_shape, stride)
    X_pad = X_pad.transpose(0, 3, 1, 2)
    X = X.transpose(0, 3, 1, 2)

    i, j, k = im2col_indices(X, kernel_shape, stride, p)
    X_col = X_pad[:, k, i, j]

    X_col = X_col.transpose(1, 2, 0).reshape(
        kernel_height * kernel_width * n_channels, -1
    )

    return X_col, p


def col2im(X_col, X, W_shape, stride, pad):
    p = pad
    n_examples, in_rows, in_cols, in_channels = X.shape
    kernel_height, kernel_width, n_in, n_out = W_shape
    kernel_shape = (kernel_height, kernel_width)

    X_pad = np.zeros(
        (n_examples, n_in, in_rows + p[0] + p[1], in_cols + p[2] + p[3])
    )
    X = X.transpose(0, 3, 1, 2)
    i, j, k = im2col_indices(X, kernel_shape, stride, pad)

    X_col_reshaped = X_col.reshape(
        n_in * kernel_height * kernel_width, -1, n_examples
    )
    X_col_reshaped = X_col_reshaped.transpose(2, 0, 1)
    np.add.at(X_pad, (slice(None), k, i, j), X_col_reshaped)

    p1 = None if p[1] == 0 else -p[1]
    p3 = None if p[3] == 0 else -p[3]

    return X_pad[:, :, p[0] : p1, p[2] : p3]


def compute_pad(X, kernel_shape, stride):
    n_examples, in_channels, in_rows, in_cols = X.shape
    s = stride

    p_rows = int(((s - 1) * in_rows - s + kernel_shape[0]) / 2)
    p_cols = int(((s - 1) * in_cols - s + kernel_shape[1]) / 2)

    return (p_rows, p_rows, p_cols, p_cols)


def pad2d(X, pad, kernel_shape, stride=None):
    p = pad
    if isinstance(p, int):
        p = (p, p, p, p)

    if isinstance(p, tuple):
        if len(p) == 2:
            p = (p[0], p[0], p[1], p[1])

        X_pad = np.pad(
            X,
            pad_width=((0, 0), (p[0], p[1]), (p[2], p[3]), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    if p == "same" and kernel_shape and stride is not None:
        p = compute_pad(X, kernel_shape, stride)
        X_pad, p = pad2d(X, p, kernel_shape, stride)
    return X_pad, p
