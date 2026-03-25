import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display
from scipy.io import wavfile
import sounddevice as sd

import time

from helper_functions.simulation_utils import convolve, generate_mic_input
from main_algorithm import nlms

# playing main source audio
# print("Near-end voice:")
# display(Audio("../data/test_signals/near_end.wav"))

# print("Far-end voice:")
# display(Audio("../data/test_signals/far_end.wav"))
fs, near_end_data = wavfile.read("../data/test_signals/near_end.wav")
_, far_end_data = wavfile.read("../data/test_signals/far_end.wav")
SAMPLE_RATE = 48000

print(" > Playing Near-end voice ...")
sd.play(near_end_data, SAMPLE_RATE)
sd.wait()
time.sleep(1)  

print(" > Playing Far-end voice ...")
sd.play(far_end_data, SAMPLE_RATE)
sd.wait()
time.sleep(1)

# converting to mono channel if stereo
if len(far_end_data.shape) > 1:
    far_end_data = np.mean(far_end_data, axis=1)

if len(near_end_data.shape) > 1:
    near_end_data = np.mean(near_end_data, axis=1)

# normalization
near_end_data = near_end_data.astype(np.float32) / np.max(np.abs(near_end_data))
far_end_data = far_end_data.astype(np.float32) / np.max(np.abs(far_end_data))


# making synthetic impulse response
FILTER_SIZE = 512
DECAY_CONSTANT = 0.003

n = np.arange(FILTER_SIZE)
# print(n[0:10])
impulse_response = np.exp(-DECAY_CONSTANT * n)

np.random.seed(42)
impulse_response *= 1 + 0.3 * np.random.randn(FILTER_SIZE)

impulse_response /= np.max(np.abs(impulse_response))
impulse_response *= 0.5
print("Impulse Response :", impulse_response[0:10], " ...")


# generating echoed far_end signal
echoed_signal = convolve(far_end_data, impulse_response)
echoed_signal = echoed_signal.astype(np.float32) / np.max(np.abs(echoed_signal))

wavfile.write('../data/results/echoed_signal.wav', SAMPLE_RATE, np.int16(echoed_signal * 32767))

_, echoed_signal_data = wavfile.read("../data/results/echoed_signal.wav")
print(" > Playing Echoed Signal ...")
sd.play(echoed_signal_data, SAMPLE_RATE)
sd.wait()
time.sleep(1)

# generating microphone input by adding near-end and echoed far-end signals
mic_input = generate_mic_input(near_end_data, echoed_signal)
mic_input = mic_input.astype(np.float32) / np.max(np.abs(mic_input))

wavfile.write('../data/results/mic_input.wav', SAMPLE_RATE, np.int16(mic_input * 32767))

print(" > Playing Microphone Input ...")
sd.play(mic_input, SAMPLE_RATE)
sd.wait()
time.sleep(1)


# applying NLMS algorithm for echo cancellation
STEP_SIZE = 0.3
REGULARIZATION = 1e-6
weights, error_signal = nlms(echoed_signal, mic_input, FILTER_SIZE, STEP_SIZE, REGULARIZATION)
error_signal = error_signal.astype(np.float32) / np.max(np.abs(error_signal))
wavfile.write('../data/results/cleaned_output.wav', SAMPLE_RATE, np.int16(error_signal * 32767))

print(" > Playing Cleaned Signal ...")
sd.play(error_signal, SAMPLE_RATE)
sd.wait()
time.sleep(1)