import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd
import time
import os

from config import FILTER_SIZE, DECAY_CONSTANT, STEP_SIZE, REGULARIZATION

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
empty_near_end = np.zeros_like(near_end_data) # to test simulation without near_end signal
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

far_end_data = far_end_data * 0.5   # The room absorbs energy, making the far-end echo quieter
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

# print(" > Playing Microphone Input ...")
# sd.play(mic_input / np.max(np.abs(mic_input)), SAMPLE_RATE)
# sd.wait()
# time.sleep(1)

##### applying NLMS algorithm for echo cancellation #####
# (STEP_SIZE, REGULARIZATION now from config.py)

# NLMS
# WITH near-end signal
mic_input_with = generate_mic_input(echoed_signal, near_end_data)
weights_with, _, error_signal_with, weights_history_with = nlms(
    far_end_data, mic_input_with, FILTER_SIZE, STEP_SIZE, REGULARIZATION
)

# WITHOUT near-end signal (near-end = 0)
mic_input_without = generate_mic_input(echoed_signal, empty_near_end)
weights_without, _, error_signal_without, weights_history_without = nlms(
    far_end_data, mic_input_without, FILTER_SIZE, STEP_SIZE, REGULARIZATION
)

# Normalizing error signal for playback and saving
# WITH near_end
error_signal_norm_with = error_signal_with.astype(np.float32) / (
    np.max(np.abs(error_signal_with)) + 1e-10
)
wavfile.write(
    "../data/results/cleaned_output_with_near_end.wav",
    SAMPLE_RATE,
    np.int16(error_signal_norm_with * 32767),
)

# WITHOUT near-end
error_signal_norm_without = error_signal_without.astype(np.float32) / (
    np.max(np.abs(error_signal_without)) + 1e-10
)
wavfile.write(
    "../data/results/cleaned_output_without_near_end.wav",
    SAMPLE_RATE,
    np.int16(error_signal_norm_without * 32767),
)
# print(" > Playing Cleaned Signal ...")
# sd.play(error_signal_norm, SAMPLE_RATE)
# sd.wait()
# time.sleep(1)

# Performance Evaluation
# Performance Evaluation - WITH near-end
L_with = len(error_signal_with)
erle_results_with = calculate_erle(echoed_signal[:L_with], error_signal_with, near_end_data[:L_with])

# Performance Evaluation - WITHOUT near-end
L_without = len(error_signal_without)
erle_results_without = calculate_erle(echoed_signal[:L_without], error_signal_without, empty_near_end[:L_without])

# Plotting WITH near-end
plt.figure(figsize=(10, 6))

# ERLE over time
plt.subplot(3, 1, 1)
plt.plot(erle_results_with, color="green", linewidth=1.5)
plt.title("Echo Return Loss Enhancement (ERLE) - WITH Near-End Signal")
plt.ylabel("ERLE (dB)")
plt.grid(True, alpha=0.3)
plt.axhline(y=15, color="orange", linestyle="--", label="15dB")
plt.axhline(y=25, color="red", linestyle="--", label="25dB")
plt.legend()

# Misalignment
plt.subplot(3, 1, 2)
plt.plot(impulse_response, label="True Room Impulse Response", linewidth=1.5)
plt.plot(weights_with, label="Estimated Weights (NLMS)", linestyle="--")
plt.title("System Identification - WITH Near-End")
plt.legend()

# MSE convergence plot
window_size = int(SAMPLE_RATE * 0.02)
stride = window_size // 2

mse_convergence_with = [
    np.mean((impulse_response - weights_history_with[i][:len(impulse_response)]) ** 2) for i in range(0, len(weights_history_with) - window_size, stride)
]

plt.subplot(3, 1, 3)
plt.plot(mse_convergence_with, color='blue', linewidth=2)
plt.title("MSE Convergence - WITH Near-End")
plt.xlabel("Window Index")
plt.ylabel("MSE")
plt.tight_layout()
plt.savefig("./../data/results/performance_metrics_with_near_end.png")
plt.show()

print(f"Final ERLE (WITH near-end): {erle_results_with[-1]:.2f} dB")

# Plotting WITHOUT near-end
plt.figure(figsize=(10, 6))

# ERLE over time
plt.subplot(3, 1, 1)
plt.plot(erle_results_without, color="green", linewidth=1.5)
plt.title("Echo Return Loss Enhancement (ERLE) - WITHOUT Near-End Signal")
plt.ylabel("ERLE (dB)")
plt.grid(True, alpha=0.3)
plt.axhline(y=15, color="orange", linestyle="--", label="15dB")
plt.axhline(y=25, color="red", linestyle="--", label="25dB")
plt.legend()

# Misalignment
plt.subplot(3, 1, 2)
plt.plot(impulse_response, label="True Room Impulse Response", linewidth=1.5)
plt.plot(weights_without, label="Estimated Weights (NLMS)", linestyle="--")
plt.title("System Identification - WITHOUT Near-End")
plt.legend()

# MSE convergence plot
mse_convergence_without = [
    np.mean((impulse_response - weights_history_without[i][:len(impulse_response)]) ** 2) for i in range(0, len(weights_history_without) - window_size, stride)
]

plt.subplot(3, 1, 3)
plt.plot(mse_convergence_without, color='blue', linewidth=2)
plt.title("MSE Convergence - WITHOUT Near-End")
plt.xlabel("Window Index")
plt.ylabel("MSE")
plt.tight_layout()
plt.savefig("./../data/results/performance_metrics_without_near_end.png")
plt.show()

print(f"Final ERLE (WITHOUT near-end): {erle_results_without[-1]:.2f} dB")