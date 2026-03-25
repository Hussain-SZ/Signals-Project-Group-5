import numpy as np


# main algorithm for adaptative filtering
def nlms(x, d, N, mu, delta):
    """
    x : np.array - Far-end reference signal
    d : np.array - Microphone (desired) signal
    N : int - Filter length (number of taps)
    mu : float - Step size parameter (0 < mu < 2)
    delta : float - Regularization parameter
    """
    w = np.zeros(N)
    e_signal = []

    common_len = min(len(x), len(d))
    for i in range(common_len):
        start = max(0, i - N + 1)
        end = i + 1
        buffer = x[start:end][::-1]

        n_zeros = N - len(buffer)
        if n_zeros > 0:
            buffer = np.concatenate([buffer, np.zeros(n_zeros)])

        y_hat = np.dot(w, buffer)
        e = d[i] - y_hat
        e_signal.append(e)

        norm_x = np.dot(buffer, buffer)
        w += (mu * e * buffer) / (norm_x + delta)

    return w, np.array(e_signal)
