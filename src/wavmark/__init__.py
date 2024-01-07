from .utils import wm_add_util, file_reader, wm_decode_util, my_parser, metric_util, path_util
from .models import my_model
import torch
import numpy as np
from huggingface_hub import hf_hub_download


def load_model(path="default"):
    if path == "default":
        resume_path = hf_hub_download(repo_id="M4869/WavMark",
                                      filename="step59000_snr39.99_pesq4.35_BERP_none0.30_mean1.81_std1.81.model.pkl",
                                      )
    else:
        resume_path = path
    model = my_model.Model(16000, num_bit=32, n_fft=1000, hop_length=400, num_layers=8)
    checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
    model_ckpt = checkpoint
    model.load_state_dict(model_ckpt, strict=True)
    model.eval()
    return model


def encode_watermark(model, signal, payload, pattern_bit_length=16, min_snr=20, max_snr=38, show_progress=False):
    device = next(model.parameters()).device

    pattern_bit = wm_add_util.fix_pattern[0:pattern_bit_length]

    watermark = np.concatenate([pattern_bit, payload])
    assert len(watermark) == 32
    signal_wmd, info = wm_add_util.add_watermark(watermark, signal, 16000, 0.1,
                                                 device, model, min_snr, max_snr,
                                                 show_progress=show_progress)
    info["snr"] = metric_util.signal_noise_ratio(signal, signal_wmd)
    return signal_wmd, info


def decode_watermark(model, signal, decode_batch_size=10, len_start_bit=16, show_progress=False):
    device = next(model.parameters()).device
    start_bit = wm_add_util.fix_pattern[0:len_start_bit]
    mean_result, info = wm_decode_util.extract_watermark_v3_batch(
        signal,
        start_bit,
        0.1,
        16000,
        model,
        device, decode_batch_size, show_progress=show_progress)

    if mean_result is None:
        return None, info

    payload = mean_result[len_start_bit:]
    return payload, info
