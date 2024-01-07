import torch.optim
import torch.nn as nn
from ..models.hinet import Hinet
import numpy as np
import random


class Model(nn.Module):
    def __init__(self, num_point, num_bit, n_fft, hop_length, num_layers):
        super(Model, self).__init__()
        self.hinet = Hinet(num_layers=num_layers)
        self.watermark_fc = torch.nn.Linear(num_bit, num_point)
        self.watermark_fc_back = torch.nn.Linear(num_point, num_bit)
        self.n_fft = n_fft
        self.hop_length = hop_length

    def stft(self, data):
        window = torch.hann_window(self.n_fft).to(data.device)
        # torch: return_complex=False is deprecDeprecated since version 2.0: return_complex=False is deprecated,
        # instead use return_complex=True Note that calling torch.view_as_real() on the output will recover the deprecated output format.
        tmp = torch.stft(data, n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=True)
        tmp = torch.view_as_real(tmp)
        # [1, 501, 41, 2]
        return tmp

    def istft(self, signal_wmd_fft):
        window = torch.hann_window(self.n_fft).to(signal_wmd_fft.device)
        # torch: return_complex=False is deprecDeprecated since version 2.0: return_complex=False is deprecated,
        # instead use return_complex=True Note that calling torch.view_as_real() on the output will recover the deprecated output format.
        return torch.istft(torch.view_as_complex(signal_wmd_fft), n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=False)

    def encode(self, signal, message):
        signal_fft = self.stft(signal)
        # (batch,freq_bins,time_frames,2)

        message_expand = self.watermark_fc(message)
        message_fft = self.stft(message_expand)

        signal_wmd_fft, msg_remain = self.enc_dec(signal_fft, message_fft, rev=False)
        # (batch,freq_bins,time_frames,2)
        signal_wmd = self.istft(signal_wmd_fft)
        return signal_wmd

    def decode(self, signal):
        signal_fft = self.stft(signal)
        watermark_fft = signal_fft
        _, message_restored_fft = self.enc_dec(signal_fft, watermark_fft, rev=True)
        message_restored_expanded = self.istft(message_restored_fft)
        message_restored_float = self.watermark_fc_back(message_restored_expanded).clamp(-1, 1)
        return message_restored_float

    def enc_dec(self, signal, watermark, rev):
        signal = signal.permute(0, 3, 2, 1)
        # [4, 2, 41, 501]
        watermark = watermark.permute(0, 3, 2, 1)
        signal2, watermark2 = self.hinet(signal, watermark, rev)
        return signal2.permute(0, 3, 2, 1), watermark2.permute(0, 3, 2, 1)
