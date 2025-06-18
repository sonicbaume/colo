import torch
from collections import OrderedDict

def convert_checkpoints(ckpt_path, device: str):
    pl_ckpt = torch.load(ckpt_path, map_location=device)
    pl_ckpt = pl_ckpt['state_dict']

    assert next(iter(pl_ckpt)).split('.')[0] == 'model', \
        'This function is to convert pl.LightningModule checkpoints to nn.Module checkpoints of the model.'

    ckpt = {k.split('.', 1)[1]: v for k, v in pl_ckpt.items()}
    ckpt = OrderedDict(ckpt)
    return ckpt
