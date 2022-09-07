import copy
import pickle
from argparse import Namespace

import torch

from utils.networks import Generator


def load_g(file_path, device):
    with open(file_path, 'rb') as f:
        old_G = pickle.load(f)['G_ema'].to(device).eval()
        old_G = old_G.float()
    return old_G

def load_from_pkl_model(tuned):
    model_state = {'init_args': tuned.init_args, 'init_kwargs': tuned.init_kwargs
        , 'state_dict': tuned.state_dict()}
    gen = Generator(*model_state['init_args'], **model_state['init_kwargs'])
    gen.load_state_dict(model_state['state_dict'])
    gen = gen.eval().cuda().requires_grad_(False)
    return gen


# def load_generators(run_id):
#     tuned, pivots, quads = load_tuned_G(run_id=run_id)
#     original = load_old_G()
#     gen = load_from_pkl_model(tuned)
#     orig_gen = load_from_pkl_model(original)
#     del tuned, original
#     return gen, orig_gen, pivots, quads
