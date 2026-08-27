"""Bypass MIOpen per gfx1151: conv1d depthwise causale come somma di shift.
MIOpen non ha i kernel CK per questa GPU e va in compilazione infinita alla
prima conv del GatedDeltaNet (misurato 3 volte, 26/8 notte). Importare questo
modulo PRIMA di qualsiasi forward transformers su GPU."""
import torch
_conv1d_vera = torch.nn.functional.conv1d
def _conv1d_senza_miopen(input=None, weight=None, bias=None, stride=1, padding=0, dilation=1, groups=1):
    x, w = input, weight
    if (x.is_cuda and groups == x.shape[1] and w.shape[1] == 1
            and stride == 1 and dilation == 1):
        K = w.shape[-1]
        pad = padding if isinstance(padding, int) else padding[0]
        xp = torch.nn.functional.pad(x, (pad, pad))
        T = xp.shape[-1] - K + 1
        y = None
        for k in range(K):
            c = xp[:, :, k:k+T] * w[:, 0, k].view(1, -1, 1)
            y = c if y is None else y + c
        if bias is not None: y = y + bias.view(1, -1, 1)
        return y
    return _conv1d_vera(x, w, bias, stride, padding, dilation, groups)
torch.nn.functional.conv1d = _conv1d_senza_miopen
