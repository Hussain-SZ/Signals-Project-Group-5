import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd
import time
import os

from config import FILTER_SIZE, DECAY_CONSTANT, STEP_SIZE, REGULARIZATION, ECHO_ATTEN

from helper_functions.simulation_utils import (
    convolve,
    generate_mic_input,
    calculate_erle,
)
from main_algorithm import nlms

# Ensure results directory exists
if not os.path.exists("../data/results"):
    os.makedirs("../data/results")

fs, near_end_data = wavfile.read("../data/test_signals/hussain-near-end.wav")
_, far_end_data = wavfile.read("../data/test_signals/hooriya-far-end.wav")

SAMPLE_RATE = fs


# print(" > Playing Near-end voice ...")
# sd.play(near_end_data, SAMPLE_RATE)
# sd.wait()
# time.sleep(1)

# print(" > Playing Far-end voice ...")
# sd.play(far_end_data, SAMPLE_RATE)
# sd.wait()
# time.sleep(1)

# converting to mono channel if stereo
if len(far_end_data.shape) > 1:
    far_end_data = np.mean(far_end_data, axis=1)

if len(near_end_data.shape) > 1:
    near_end_data = np.mean(near_end_data, axis=1)

# normalization
near_end_data = near_end_data.astype(np.float32) / np.max(np.abs(near_end_data))

far_end_data = far_end_data * ECHO_ATTEN   # The room absorbs energy, making the far-end echo quieter
far_end_data = far_end_data.astype(np.float32) / np.max(np.abs(far_end_data))
# SET TO ZERO For testing purposes
# near_end_data = np.zeros_like(near_end_data).astype(np.float32)



# making synthetic impulse response
# (FILTER_SIZE and DECAY_CONSTANT now from config.py)

n = np.arange(FILTER_SIZE)
np.random.seed(42)
# A real acoustic room echo oscillates around zero (decaying white noise)
impulse_response = np.random.randn(FILTER_SIZE) * np.exp(-DECAY_CONSTANT * n)
impulse_response *= 0.5 / np.max(np.abs(impulse_response))

print("Impulse Response :", impulse_response[0:10], " ...")

# generating echoed far_end signal
echoed_signal = convolve(far_end_data, impulse_response)[: len(far_end_data)]
# echoed_signal = echoed_signal.astype(np.float32) / np.max(np.abs(echoed_signal))

# only normalizing when writing to .wav file,
# not for algorithm input
wavfile.write(
    "../data/results/echoed_signal.wav",
    SAMPLE_RATE,
    np.int16((echoed_signal / np.max(np.abs(echoed_signal))) * 32767),
)

# _, echoed_signal_data = wavfile.read("../data/results/echoed_signal.wav")
# print(" > Playing Echoed Signal ...")
# sd.play(echoed_signal_data, SAMPLE_RATE)
# sd.wait()
# time.sleep(1)

# generating microphone input by adding near-end and echoed far-end signals
mic_input = generate_mic_input(echoed_signal, near_end_data)

# Same here, no normalization for algorithm input
# mic_input = mic_input.astype(np.float32) / np.max(np.abs(mic_input))

wavfile.write(
    "../data/results/mic_input.wav",
    SAMPLE_RATE,
    np.int16((mic_input / np.max(np.abs(mic_input))) * 32767),
)

print(" > Playing Microphone Input ...")
sd.play(mic_input / np.max(np.abs(mic_input)), SAMPLE_RATE)
sd.wait()
time.sleep(1)

##### applying NLMS algorithm for echo cancellation #####
# (STEP_SIZE, REGULARIZATION now from config.py)

# NLMS
weights, estimated_echo, error_signal = nlms(
    far_end_data, mic_input, FILTER_SIZE, STEP_SIZE, REGULARIZATION
)

# Normalizing error signal for playback and saving
error_signal_norm = error_signal.astype(np.float32) / (
    np.max(np.abs(error_signal)) + 1e-10
)
wavfile.write(
    "../data/results/cleaned_output.wav",
    SAMPLE_RATE,
    np.int16(error_signal_norm * 32767),
)

print(" > Playing Cleaned Signal ...")
sd.play(error_signal_norm, SAMPLE_RATE)
sd.wait()
time.sleep(1)

# Performance Evaluation
# ERLE calculation
L = len(error_signal)
erle_results = calculate_erle(echoed_signal[:L], error_signal, near_end_data[:L])

# Plotting
plt.figure(figsize=(10, 6))

# ERLE over time
plt.subplot(2, 1, 1)
plt.plot(erle_results, color="green", linewidth=1.5)
plt.title("Echo Return Loss Enhancement (ERLE)")
plt.ylabel("ERLE (dB)")
plt.grid(True, alpha=0.3)
plt.axhline(y=15, color="orange", linestyle="--", label="15dB")
plt.axhline(y=25, color="red", linestyle="--", label="25dB")
plt.legend()

# Misalignment
# Comparing learned weights to true synthetic impulse response
plt.subplot(2, 1, 2)
# Using plot instead of stem for cleaner view with 512 taps
plt.plot(impulse_response, label="True Room Impulse Response", linewidth=1.5)
plt.plot(weights, label="Estimated Weights (NLMS)", linestyle="--")
plt.title("System Identification")
plt.legend()

plt.tight_layout()
plt.savefig("../data/results/performance_metrics.png")
plt.show()

print(f"Final ERLE: {erle_results[-1]:.2f} dB")
