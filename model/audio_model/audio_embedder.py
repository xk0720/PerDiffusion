import torch
import torch.nn as nn


class AudioEmbedder(nn.Module):
    def __init__(self, skip_norm=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_norm = skip_norm

    def _encode(self, x):
        # x.shape: (bs, seq_len, 78)
        
        if not self.skip_norm:
            x_min = torch.min(x)
            x_max = torch.max(x)
            return (x - x_min) / (x_max - x_min)
        else:
            return x
        
        #     x_mean = torch.mean(x)
        #     x_std = torch.std(x, unbiased=True) # unbiased?
        #     return (x - x_mean) / x_std
