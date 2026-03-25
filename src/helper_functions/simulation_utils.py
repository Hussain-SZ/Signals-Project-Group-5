import numpy as np

# convolution helper function
def convolve(x, h):
    L = len(x)
    N = len(h)

    output_len = L + N - 1
    y = np.zeros(output_len)
    h_flipped = h[::-1]
    x_padded = np.concatenate([np.zeros(N-1), x, np.zeros(N-1)])

    for i in range(output_len):
        window = x_padded[i : i + N]
        y[i] = np.dot(window, h_flipped)
    
    return y

# generate microphone input audio
def generate_mic_input(x, y_echo):
    common_len = min(len(x), len(y_echo))

    d = x[:common_len] + y_echo[:common_len]

    return d
