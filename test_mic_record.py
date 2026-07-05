import argparse
import queue
import time

import numpy as np
import sounddevice as sd
from scipy.io import wavfile


def main() -> None:
    parser = argparse.ArgumentParser(description="Record and playback microphone audio.")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--samplerate", type=int, default=None)
    parser.add_argument("--outfile", default="mic_test.wav")
    args = parser.parse_args()

    device_sample_rate = args.samplerate
    if device_sample_rate is None and args.device is not None:
        device_info = sd.query_devices(args.device, "input")
        device_sample_rate = int(device_info["default_samplerate"])
        print(f"sample_rate из устройства: {device_sample_rate} Hz")
    if device_sample_rate is None:
        device_sample_rate = 44100

    frames = int(device_sample_rate * args.seconds)
    audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames_count, time_info, status) -> None:
        if status:
            print(f"audio status: {status}")
        audio_queue.put(indata.copy())

    stream_kwargs = {}
    if args.device is not None:
        stream_kwargs["device"] = args.device
        print(f"использую входное устройство: {args.device}")

    print(f"запись {args.seconds:.1f} сек...")
    with sd.InputStream(
        samplerate=device_sample_rate,
        channels=1,
        dtype="float32",
        callback=callback,
        **stream_kwargs,
    ):
        chunks = []
        remaining = frames
        while remaining > 0:
            chunk = audio_queue.get()
            chunks.append(chunk)
            remaining -= chunk.shape[0]
            time.sleep(0.01)

    audio = np.concatenate(chunks, axis=0)[:frames]
    audio_int16 = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_int16 * 32767).astype(np.int16)

    wavfile.write(args.outfile, device_sample_rate, audio_int16)
    print(f"сохранено: {args.outfile}")

    print("проигрываю запись...")
    sd.play(audio, device_sample_rate)
    sd.wait()
    print("готово.")


if __name__ == "__main__":
    main()
