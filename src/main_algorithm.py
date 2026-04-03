import numpy as np


def nlms(x, d, filter_size, step_size, regularization):
    """
    x = far-end reference signal
    d = microphone signal
    """

    x = np.asarray(x, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)

    length = min(len(x), len(d))
    x = x[:length]
    d = d[:length]

    weights = np.zeros(filter_size, dtype=np.float64)
    error_signal = np.zeros(length, dtype=np.float64)
    estimated_echo = np.zeros(length, dtype=np.float64)

    for n in range(length):
        x_vec = np.zeros(filter_size, dtype=np.float64)

        start = max(0, n - filter_size + 1)
        current_samples = x[start:n + 1][::-1]
        x_vec[:len(current_samples)] = current_samples

        y_hat = np.dot(weights, x_vec)
        error = d[n] - y_hat

        estimated_echo[n] = y_hat
        error_signal[n] = error

        x_max = np.max(np.abs(x_vec)) 
        if np.abs(d[n]) > 0.5 * x_max:
            # Double-talk detected, skip weight update
            continue

        norm = np.dot(x_vec, x_vec) + regularization
        weights = weights + (step_size / norm) * error * x_vec

    return weights, estimated_echo, error_signal
    
