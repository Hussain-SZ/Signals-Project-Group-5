"""
Live demo: far-end from a WAV file, near-end from the microphone.
Phases 0–3 match the course script; outputs go to ../data/demo/

Audio is captured in real time; echo cancellation uses `nlms` from
`main_algorithm` on the recorded buffers (cumulative across phases when
weights are carried, matching streaming continuity).

Run from the src folder:
  python live_demo.py

Needs: sounddevice, numpy, scipy, tqdm
  pip install sounddevice numpy scipy tqdm

If cancellation sounds wrong: try REF_DELAY_SAMPLES > 0 (in samples at the *working* rate).
"""

import json
import os
import sys
import time

import numpy as np
import sounddevice as sd
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm

from config import FILTER_SIZE, REGULARIZATION, STEP_SIZE

from main_algorithm import nlms

FAR_END_WAV = "../data/test_signals/test_far_end.wav"
OUT_DIR = "../data/demo"

PHASE_DURATION_SEC = 40.0

INPUT_DEVICE = None
OUTPUT_DEVICE = None
FORCE_SAMPLE_RATE = None

BLOCK_SIZE = 1024

REF_DELAY_SAMPLES = 0
GEIGEL_RATIO_IN_MAIN = 0.5

PHASES = [
    ("phase0_baseline", False, False, False),
    ("phase1_convergence", True, False, True),
    ("phase2_double_talk", True, True, False),
    ("phase3_path_change", True, True, False),
]


def _duplex_device():
    if INPUT_DEVICE is None and OUTPUT_DEVICE is None:
        return None
    d = sd.default.device
    if isinstance(d, (list, tuple)) and len(d) >= 2:
        din, dout = d[0], d[1]
    else:
        din = dout = d
    din = INPUT_DEVICE if INPUT_DEVICE is not None else din
    dout = OUTPUT_DEVICE if OUTPUT_DEVICE is not None else dout
    return (din, dout)


def _query_default_sr_for_duplex(duplex):
    """Sample rate the output side of the duplex path prefers (float or None)."""
    try:
        if duplex is None:
            dev = sd.query_devices(kind="output")
        else:
            dev = sd.query_devices(duplex[1])
        sr = dev.get("default_samplerate")
        if sr and sr > 0:
            return float(sr)
    except Exception:
        pass
    return None


def choose_working_sample_rate(fs_wav, duplex):
    """
    Pick one rate for playback + capture + saved WAVs.
    Prefer device default; optionally FORCE_SAMPLE_RATE; else fall back to WAV rate.
    """
    if FORCE_SAMPLE_RATE is not None:
        fs = float(FORCE_SAMPLE_RATE)
        print("Using forced sample rate:", int(fs), "Hz")
        return fs

    sr_dev = _query_default_sr_for_duplex(duplex)
    if sr_dev:
        if abs(sr_dev - fs_wav) > 0.5:
            print(
                "Device default rate",
                int(sr_dev),
                "Hz differs from WAV",
                fs_wav,
                "Hz — resampling far-end to",
                int(sr_dev),
                "Hz.",
            )
        else:
            print("WAV rate matches device default:", int(sr_dev), "Hz")
        return sr_dev

    print("Could not read device default rate; using WAV sample rate:", fs_wav, "Hz")
    return float(fs_wav)


def resample_far_end(far_end, fs_from, fs_to):
    """Resample far-end to fs_to (Hz) for smooth playback at the stream rate."""
    if abs(fs_from - fs_to) < 0.5:
        return np.asarray(far_end, dtype=np.float32)
    new_len = max(2, int(round(len(far_end) * fs_to / fs_from)))
    y = signal.resample(np.asarray(far_end, dtype=np.float64), new_len)
    peak = np.max(np.abs(y)) + 1e-12
    return (y / peak).astype(np.float32)


def print_audio_devices():
    print("--- Audio devices (check mic + speakers are what you expect) ---")
    try:
        print(sd.query_devices())
    except Exception as ex:
        print("Could not list devices:", ex)
    print("Default device pair (input, output):", sd.default.device)
    print("Using duplex device:", _duplex_device() or "(PortAudio default)")
    print("---------------------------------------------------------------")


class DelayLine:
    def __init__(self, delay_samples):
        self.D = int(delay_samples)
        if self.D == 0:
            self._buf = None
        else:
            self._buf = np.zeros(self.D, dtype=np.float64)
            self._idx = 0

    def push(self, x):
        if self.D == 0:
            return float(x)
        out = float(self._buf[self._idx])
        self._buf[self._idx] = x
        self._idx = (self._idx + 1) % self.D
        return out


def load_far_end_mono(path):
    fs, data = wavfile.read(path)
    data = np.asarray(data, dtype=np.float64)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    peak = np.max(np.abs(data)) + 1e-12
    data = (data / peak).astype(np.float32)
    return int(fs), data


def write_wav_int16(path, fs, x):
    x = np.asarray(x, dtype=np.float32).ravel()
    p = np.max(np.abs(x)) + 1e-12
    x = (x / p * 0.98).clip(-1.0, 1.0)
    pcm = np.int16(x * 32767)
    wavfile.write(path, int(round(fs)), pcm)


class PhaseStreamRunner:
    """Duplex callback: play far-end; record mic + aligned reference for offline NLMS."""

    def __init__(self):
        self.pos = 0

    def make_callback(
        self,
        delay_line,
        far_end,
        n_samples,
        mic_out,
        ref_out,
    ):
        n_far = len(far_end)
        far_idx = 0

        def callback(indata, outdata, frames, _time, status):
            nonlocal far_idx
            if status:
                print(status, file=sys.stderr)
            if self.pos >= n_samples:
                outdata.fill(0)
                raise sd.CallbackStop
            for i in range(frames):
                if self.pos >= n_samples:
                    outdata[i:, 0] = 0
                    raise sd.CallbackStop
                x_play = float(far_end[far_idx % n_far])
                far_idx += 1
                outdata[i, 0] = x_play
                d = float(indata[i, 0])
                x_ref = delay_line.push(x_play)
                mic_out[self.pos] = d
                ref_out[self.pos] = x_ref
                self.pos += 1

        return callback


def run_phase_duplex_stream(
    delay_line,
    far_end,
    fs,
    block_size,
    n_samples,
    duplex,
    phase_name,
    phase_duration_sec,
):
    mic_out = np.zeros(n_samples, dtype=np.float64)
    ref_out = np.zeros(n_samples, dtype=np.float64)

    runner = PhaseStreamRunner()

    stream_kw = dict(
        samplerate=fs,
        blocksize=block_size,
        dtype="float32",
        channels=1,
        latency="low",
        callback=runner.make_callback(delay_line, far_end, n_samples, mic_out, ref_out),
    )
    if duplex is not None:
        stream_kw["device"] = duplex

    stream = sd.Stream(**stream_kw)

    with tqdm(
        total=phase_duration_sec,
        unit="s",
        desc=phase_name,
        bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f}s [{elapsed}<{remaining}]",
    ) as pbar:
        with stream:
            while runner.pos < n_samples:
                time.sleep(0.05)
                p = runner.pos
                pbar.n = min(p / float(fs), phase_duration_sec)
                pbar.refresh()
            pbar.n = phase_duration_sec

    return mic_out, ref_out


def compute_phase_metrics(
    error,
    mic,
    final_weight_l2,
    nlms_hist_len,
    hist_len_prev,
    phase_len,
    phase_id,
    name,
    duration_s,
    adaptation,
    geigel,
):
    e = error.astype(np.float64)
    m = mic.astype(np.float64)
    eps = 1e-18
    rms_e = float(np.sqrt(np.mean(e**2)))
    rms_m = float(np.sqrt(np.mean(m**2)))
    cancel_db = 10.0 * np.log10((np.mean(m**2) + eps) / (np.mean(e**2) + eps))
    u = nlms_hist_len - hist_len_prev
    fz = int(phase_len - u)
    return {
        "phase_id": phase_id,
        "name": name,
        "duration_s": duration_s,
        "adaptation": adaptation,
        "geigel": geigel,
        "rms_mic": rms_m,
        "rms_error": rms_e,
        "mic_to_error_power_db": float(cancel_db),
        "weight_l2": float(final_weight_l2),
        "double_talk_freeze_samples": fz,
        "nlms_update_samples": int(u),
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    far_path = os.path.normpath(FAR_END_WAV)
    out_dir = os.path.normpath(OUT_DIR)

    if not os.path.isfile(far_path):
        print("Could not find far-end WAV:", far_path)
        sys.exit(1)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print_audio_devices()

    fs_wav, far_raw = load_far_end_mono(far_path)
    print("WAV file sample rate:", fs_wav, "Hz")

    duplex = _duplex_device()
    fs = choose_working_sample_rate(fs_wav, duplex)
    far_end = resample_far_end(far_raw, fs_wav, fs)
    print("Far-end length (samples):", len(far_end), "at working rate", int(fs), "Hz")

    n_samples = int(round(PHASE_DURATION_SEC * fs))
    duration_s = float(n_samples) / fs

    session_error = []
    session_mic = []
    all_metrics = []

    cum_ref_parts = []
    cum_mic_parts = []
    hist_len_prev = 0

    for phase_id, row in enumerate(PHASES):
        name, adaptation, geigel, reset_weights = row

        print("\n" + "=" * 60)
        print("Phase:", name, "|", round(duration_s, 2), "s audio @", int(fs), "Hz")
        print("  adaptation =", adaptation, "  geigel =", geigel)
        if phase_id == 0:
            print("  (Raw echo — process with identity for this phase.)")
        elif phase_id == 1:
            print("  (Stay silent — convergence.)")
        elif phase_id == 2:
            print("  (Speak — double-talk.)")
        else:
            print("  (Change room — path change.)")

        if reset_weights:
            cum_ref_parts.clear()
            cum_mic_parts.clear()
            hist_len_prev = 0

        delay_line = DelayLine(REF_DELAY_SAMPLES)

        mic, ref = run_phase_duplex_stream(
            delay_line,
            far_end,
            fs,
            BLOCK_SIZE,
            n_samples,
            duplex,
            name,
            duration_s,
        )

        if not adaptation:
            err = mic.astype(np.float64).copy()
            final_w_norm = 0.0
            wh_len = 0
            # No NLMS; freeze/update counts not applicable for this phase
            m = compute_phase_metrics(
                err,
                mic,
                final_w_norm,
                0,
                0,
                n_samples,
                phase_id,
                name,
                duration_s,
                adaptation,
                geigel,
            )
            m["nlms_update_samples"] = 0
            m["double_talk_freeze_samples"] = 0
        else:
            cum_ref_parts.append(ref.astype(np.float64))
            cum_mic_parts.append(mic.astype(np.float64))
            R = np.concatenate(cum_ref_parts)
            M = np.concatenate(cum_mic_parts)
            weights, _, err_full, weights_history = nlms(
                R, M, FILTER_SIZE, STEP_SIZE, REGULARIZATION
            )
            off = sum(len(x) for x in cum_ref_parts[:-1])
            err = err_full[off : off + n_samples].copy()
            final_w_norm = float(np.linalg.norm(weights))
            wh_len = len(weights_history)

            m = compute_phase_metrics(
                err,
                mic,
                final_w_norm,
                wh_len,
                hist_len_prev,
                n_samples,
                phase_id,
                name,
                duration_s,
                adaptation,
                geigel,
            )
            hist_len_prev = wh_len

        session_error.append(err)
        session_mic.append(mic)

        wav_err = os.path.join(out_dir, name + "_error.wav")
        wav_mic = os.path.join(out_dir, name + "_mic.wav")
        write_wav_int16(wav_err, fs, err)
        write_wav_int16(wav_mic, fs, mic)

        all_metrics.append(m)
        with open(os.path.join(out_dir, name + "_metrics.json"), "w") as f:
            json.dump(m, f, indent=2)

        print("Saved", os.path.basename(wav_err), ",", os.path.basename(wav_mic))

    if session_error:
        full_err = np.concatenate(session_error)
        full_mic = np.concatenate(session_mic)
        write_wav_int16(os.path.join(out_dir, "session_full_error.wav"), fs, full_err)
        write_wav_int16(os.path.join(out_dir, "session_full_mic.wav"), fs, full_mic)

    summary = {
        "wav_file_sample_rate_hz": fs_wav,
        "working_sample_rate_hz": fs,
        "force_sample_rate": FORCE_SAMPLE_RATE,
        "far_end_file": far_path,
        "phase_duration_sec": PHASE_DURATION_SEC,
        "input_device": INPUT_DEVICE,
        "output_device": OUTPUT_DEVICE,
        "filter_size": FILTER_SIZE,
        "step_size": STEP_SIZE,
        "regularization": REGULARIZATION,
        "geigel_ratio_matches_main_algorithm": GEIGEL_RATIO_IN_MAIN,
        "ref_delay_samples": REF_DELAY_SAMPLES,
        "block_size": BLOCK_SIZE,
        "phases": all_metrics
    }
    with open(os.path.join(out_dir, "session_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone. See", os.path.join(out_dir, "session_metrics.json"))


if __name__ == "__main__":
    main()
