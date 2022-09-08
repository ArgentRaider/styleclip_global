import argparse
import math
import os
import pickle
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from utils.networks import SynthesisBlock


def get_affine_layers(synthesis):
    blocks: List[SynthesisBlock] = [getattr(synthesis, f'b{res}') for res in synthesis.block_resolutions]
    affine_layers = []
    for block in blocks:
        if hasattr(block, 'conv0'):
            affine_layers.append((block.conv0.affine, True))
        affine_layers.append((block.conv1.affine, True))
        affine_layers.append((block.torgb.affine, False))
    return affine_layers


def to_styles(edit: torch.Tensor, affine_layers):
    '''
        Divide a long vector into different layers of StyleSpace.
    '''
    idx = 0
    styles = []
    for layer, is_conv in affine_layers:
        layer_dim = layer.weight.shape[0]
        if is_conv:
            styles.append(edit[idx:idx + layer_dim].clone())
            idx += layer_dim
        else:
            styles.append(torch.zeros(layer_dim, device=edit.device, dtype=edit.dtype))

    return styles

def vec_to_styles(edit: torch.Tensor, affine_layers):
    '''
        Divide a long vector (or a batch) into different layers of StyleSpace.
        Compared to 'to_styles', each layer has an additional 'batch_size' dimension.
    '''
    idx = 0
    styles = []
    for layer, is_conv in affine_layers:
        layer_dim = layer.weight.shape[0]
        if is_conv:
            if len(edit.shape) == 1:
                styles.append(edit[idx:idx + layer_dim].clone().unsqueeze(0))
            elif len(edit.shape) == 2:
                styles.append(edit[:, idx:idx + layer_dim].clone())
            else:
                print("Invalid shape", edit.shape, ", input tensor 'edit' should be 1-dim of 2-dim.")
                exit(-1)
            idx += layer_dim
        else:
            if len(edit.shape) == 1:
                styles.append(torch.zeros(layer_dim, device=edit.device, dtype=edit.dtype).unsqueeze(0))
            elif len(edit.shape) == 2:
                styles.append(torch.zeros((edit.shape[0], layer_dim), device=edit.device, dtype=edit.dtype))
            else:
                print("Invalid shape", edit.shape, ", input tensor 'edit' should be 1-dim of 2-dim.")
                exit(-1)

    return styles

def styles_to_vec(styles, affine_layers):
    '''
        Reverse operation of 'vec_to_styles'
    '''
    res = None
    idx = 0
    for layer, is_conv in affine_layers:
        if is_conv:
            style = styles[idx]
            if res is None:
                res = style.clone()
            else:
                res = torch.hstack((res, style))
        idx += 1
    return res


def w_to_styles(w, affine_layers):
    '''
        Transfer W/W+ vectors to StyleSpace vectors.
    '''
    w_idx = 0
    styles = []
    for affine, is_conv in affine_layers:
        styles.append(affine(w[:, w_idx]))
        if is_conv:
            w_idx += 1

    return styles