import torch
import numpy as np


def calc_ber(watermark_decoded_tensor, watermark_tensor, threshold=0.5):
    watermark_decoded_binary = watermark_decoded_tensor >= threshold
    watermark_binary = watermark_tensor >= threshold
    ber_tensor = 1 - (watermark_decoded_binary == watermark_binary).to(torch.float32).mean()
    return ber_tensor


def to_equal_length(original, signal_watermarked):
    if original.shape != signal_watermarked.shape:
        print("Warning: length not equal:", len(original), len(signal_watermarked))
        min_length = min(len(original), len(signal_watermarked))
        original = original[0:min_length]
        signal_watermarked = signal_watermarked[0:min_length]
    assert original.shape == signal_watermarked.shape
    return original, signal_watermarked


def signal_noise_ratio(original, signal_watermarked):
    original, signal_watermarked = to_equal_length(original, signal_watermarked)
    noise_strength = np.sum((original - signal_watermarked) ** 2)
    if noise_strength == 0:  #
        return np.inf
    signal_strength = np.sum(original ** 2)
    ratio = signal_strength / noise_strength
    ratio = max(1e-10, ratio)
    return 10 * np.log10(ratio)


def batch_signal_noise_ratio(original, signal_watermarked):
    signal = original.detach().cpu().numpy()
    signal_watermarked = signal_watermarked.detach().cpu().numpy()
    tmp_list = []
    for s, swm in zip(signal, signal_watermarked):
        out = signal_noise_ratio(s, swm)
        tmp_list.append(out)
    return np.mean(tmp_list)


def resample_to16k(data, old_sr):
    new_fs = 16000
    new_data = data[::int(old_sr / new_fs)]
    return new_data
