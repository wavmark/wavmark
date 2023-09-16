import torch
import numpy as np
from ..utils import metric_util
import tqdm
import time

# The pattern bits can be any random sequence.
# But don't use all-zeros, all-ones, or any periodic sequence, which will seriously hurt decoding performance.
fix_pattern = [1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0,
               0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1,
               1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1,
               1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0,
               0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0]


def add_watermark(bit_arr, data, num_point, shift_range, device, model, min_snr, max_snr, show_progress):
    t1 = time.time()

    chunk_size = num_point + int(num_point * shift_range)
    num_segments = int(len(data) / chunk_size)
    len_remain = len(data) - num_segments * chunk_size

    output_chunks = []
    encoded_sections = 0
    skip_sections = 0

    the_iter = range(num_segments)
    if show_progress:
        the_iter = tqdm.tqdm(the_iter, desc="Processing")

    for i in the_iter:
        start_point = i * chunk_size
        current_chunk = data[start_point:start_point + chunk_size].copy()
        # [watermark_segment | shift_area ]
        current_chunk_cover_area = current_chunk[0:num_point]
        current_chunk_shift_area = current_chunk[num_point:]
        current_chunk_cover_area_wmd, state = encode_trunck_with_snr_check(i, current_chunk_cover_area,
                                                                           bit_arr,
                                                                           device, model, min_snr, max_snr)

        if state == "skip":
            skip_sections += 1
        else:
            encoded_sections += 1

        output = np.concatenate([current_chunk_cover_area_wmd, current_chunk_shift_area])
        assert output.shape == current_chunk.shape
        output_chunks.append(output)

    assert len(output_chunks) > 0
    if len_remain > 0:
        output_chunks.append(data[len(data) - len_remain:])

    reconstructed_array = np.concatenate(output_chunks)

    time_cost = time.time() - t1

    info = {
        "time_cost": time_cost,
        "encoded_sections": encoded_sections,
        "skip_sections": skip_sections,
    }
    return reconstructed_array, info


def encode_trunck_with_snr_check(idx_trunck, signal, wm, device, model, min_snr, max_snr):
    signal_for_encode = signal
    encode_times = 0
    while True:
        encode_times += 1
        signal_wmd = encode_trunck(signal_for_encode, wm, device, model)
        snr = metric_util.signal_noise_ratio(signal, signal_wmd)
        if encode_times == 1 and snr < min_snr:
            print("skip section:%d, snr too low:%.1f" % (idx_trunck, min_snr))
            return signal, "skip"

        if snr < max_snr:
            return signal_wmd, encode_times
        # snr is too hugh
        signal_for_encode = signal_wmd

        if encode_times > 10:
            return signal_wmd, encode_times


def encode_trunck(trunck, wm, device, model):
    with torch.no_grad():
        signal = torch.FloatTensor(trunck).to(device)[None]
        message = torch.FloatTensor(np.array(wm)).to(device)[None]
        signal_wmd_tensor = model.encode(signal, message)
        signal_wmd = signal_wmd_tensor.detach().cpu().numpy().squeeze()
        return signal_wmd
