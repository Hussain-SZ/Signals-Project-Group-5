import numpy as np

# main algorithm for adaptive filter
def nlms(y, d, N, mu, delta):
    w = np.zeros(N)
    errors = []

    common_len = min(len(y), len(d))
    for i in range(common_len):
        start = max(0, i - N + 1)
        end = i + 1
        buffer = y[start:end][::-1]

        n_zeros = N - len(buffer)
        if n_zeros > 0:
            buffer = np.concatenate([buffer, np.zeros(n_zeros)])
        
        y_echo = np.dot(w, buffer)
        error = d[i] - y_echo
        errors.append(error)
        w += (mu * error * buffer)/(np.dot(buffer,buffer) + delta)
    
    return w, np.array(errors)