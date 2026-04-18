import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from main_algorithm import nlms
from helper_functions.simulation_utils import resample_audio
from config import FILTER_SIZE, STEP_SIZE, REGULARIZATION
import os
import matplotlib.pyplot as plt


TARGET_FS = 16000
BLOCK_SIZE = 512
PHASE = 3  # 0: No adaptation, >0 Adaptation

# Resample
fs_orig, far_end_raw = wavfile.read("../data/test_signals/test-far-end.wav")
far_end = resample_audio(far_end_raw, fs_orig, TARGET_FS)
far_end = far_end.astype(np.float32) / (np.max(np.abs(far_end)) + 1e-10)

weights = np.zeros(FILTER_SIZE)
cleaned_audio_storage = []
pointer = 0
current_step = 0 if PHASE == 0 else STEP_SIZE

DELAY_SAMPLES = 3955 
x_history = np.zeros(DELAY_SAMPLES + BLOCK_SIZE, dtype=np.float32)

print(f"--- RUNNING PHASE {PHASE} ---")
print("Press Ctrl+C to stop the demo and save the recording.")

erle_history = []
mse_history = []

# Real-Time Loop
with sd.OutputStream(
    samplerate=TARGET_FS, channels=1, dtype="float32"
) as out_stream, sd.InputStream(
    samplerate=TARGET_FS, channels=1, dtype="float32"
) as in_stream:

    out_stream.start()
    in_stream.start()

    try:
        while True:  # Changed to infinite loop for looping audio
            # Reset pointer if we hit the end of the file to loop the audio
            if pointer + BLOCK_SIZE >= len(far_end):
                pointer = 0

            x_block = far_end[pointer : pointer + BLOCK_SIZE].astype(np.float32)

            # Write to speakers & read from mic
            out_stream.write(x_block)
            d_block, _ = in_stream.read(BLOCK_SIZE)
            d_block = d_block.flatten()
            # print(d_block.ndim, d_block.shape)

            # Update delay buffer
            x_history = np.roll(x_history, -BLOCK_SIZE)
            x_history[-BLOCK_SIZE:] = x_block
            delayed_x_block = x_history[:BLOCK_SIZE]
            # input_to_nlms = x_history[-(BLOCK_SIZE + FILTER_SIZE):]

            # NLMS Algorithm
            w_out, _, e_block, _ = nlms(
                delayed_x_block, d_block, FILTER_SIZE, current_step, REGULARIZATION, weights
            )
            if PHASE != 0:
                weights = w_out

            # ERLE: Ratio of Mic Input Power to Error Signal Power
            p_d = np.sum(d_block**2) + 1e-10
            p_e = np.sum(e_block**2) + 1e-10
            erle_history.append(10 * np.log10(p_d / p_e))

            # MSE of the error signal
            mse_history.append(np.mean(e_block**2))

            cleaned_audio_storage.extend(e_block)
            pointer += BLOCK_SIZE

    except KeyboardInterrupt:
        print("\nStopping demo and saving recording...")

if erle_history:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

    # ERLE plot
    ax1.plot(erle_history, color='steelblue', linewidth=1, label="Smoothed ERLE (dB)")
    ax1.axhline(y=15, color="orange", linestyle="--", label="15 dB (Acceptable)")
    ax1.axhline(y=25, color="red",    linestyle="--", label="25 dB (Excellent)")
    ax1.set_title(f"Real-Time ERLE — Phase {PHASE}")
    ax1.set_xlabel("Block Index")
    ax1.set_ylabel("ERLE (dB)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # FIX #7: MSE plot — now actually plotted
    ax2.plot(mse_history, color='tomato', linewidth=1, label="MSE (error signal)")
    ax2.set_title("MSE of Error Signal Over Time")
    ax2.set_xlabel("Block Index")
    ax2.set_ylabel("MSE")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # log scale makes convergence much more visible

    plt.tight_layout()
    plt.savefig(f"../data/demo/metrics_phase_{PHASE}.png", dpi=150)
    plt.show()

    print(f"Median ERLE: {np.median(erle_history):.1f} dB")
    print(f"Final  ERLE: {erle_history[-1]:.1f} dB")


if cleaned_audio_storage:
    # Fixed the path to point to your demo directory
    out_path = f"../data/demo/demo_phase_{PHASE}.wav"
    wavfile.write(
        out_path, TARGET_FS, np.array(cleaned_audio_storage).astype(np.float32)
    )
    print(f"Saved recording to: {out_path}")
