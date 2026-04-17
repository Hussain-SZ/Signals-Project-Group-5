import numpy as np

def nlms(x, d, filter_size, step_size, regularization, initial_weights=None):
    """
    x = far-end reference signal
    d = microphone signal
    initial_weights = persistent weights from previous block (used in real-time)
    """
    x = np.asarray(x, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)

    length = min(len(x), len(d))
    x = x[:length]
    d = d[:length]

    if initial_weights is not None:
        weights = initial_weights.copy()
    else:
        weights = np.zeros(filter_size, dtype=np.float64)

    error_signal = np.zeros(length, dtype=np.float64)
    estimated_echo = np.zeros(length, dtype=np.float64)
    weights_history = []

    for n in range(length):
        x_vec = np.zeros(filter_size, dtype=np.float64)
        start = max(0, n - filter_size + 1)
        current_samples = x[start:n + 1][::-1]
        x_vec[:len(current_samples)] = current_samples

        y_hat = np.dot(weights, x_vec)
        error = d[n] - y_hat

        estimated_echo[n] = y_hat
        error_signal[n] = error

        # Double-talk detection 
        x_max = np.max(np.abs(x_vec)) + 1e-10
        if np.abs(d[n]) > 0.5 * x_max:
            continue

        norm = np.dot(x_vec, x_vec) + regularization
        weights = weights + (step_size / norm) * error * x_vec
        weights_history.append(weights.copy())

    return weights, estimated_echo, error_signal, weights_history