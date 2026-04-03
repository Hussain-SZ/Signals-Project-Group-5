import numpy as np


# convolution helper function
def convolve(x, h):
    L = len(x)
    N = len(h)

    output_len = L + N - 1
    y = np.zeros(output_len)
    h_flipped = h[::-1]
    x_padded = np.concatenate([np.zeros(N - 1), x, np.zeros(N - 1)])

    for i in range(output_len):
        window = x_padded[i : i + N]
        y[i] = np.dot(window, h_flipped)

    return y[:len(x)]


# generate microphone input audio
def generate_mic_input(y_echo, s, noise_level=0.001):
    common_len = min(len(y_echo), len(s))
    noise = noise_level * np.random.randn(common_len)
    # Mic = Echo + Near-end + Noise 
    # Echo is the far-end signal convolved with the room impulse response
    # Near-end is the local speech signal (initially 0) 
    # Weak random noise is added
    d = y_echo[:common_len] + s[:common_len] + noise 
    return d


# ERLE calculation over sliding window
def calculate_erle(echo, error, near_end, window_size=2000):
    erle_points = []
    eps = 1e-6  # avoid division by zero

    # Process in windows to see performance over time
    for i in range(0, len(error) - window_size, window_size):
        echo_w = echo[i : i + window_size]
        # residual echo = error signal - actual near-end speech
        residual_w = error[i : i + window_size] - near_end[i : i + window_size]

        p_echo = np.sum(echo_w**2)
        p_residual = np.sum(residual_w**2)

        # ERLE formula
        erle = 10 * np.log10(p_echo / (p_residual + eps))
        erle_points.append(erle)

    return np.array(erle_points)
