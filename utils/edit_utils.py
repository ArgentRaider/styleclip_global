import argparse
import math
import os
import pickle
from typing import Tuple, List, Union

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
        Divide a long edit(offset) vector into different layers of StyleSpace.
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

def vec_to_styles(vec: torch.Tensor, affine_layers) -> List:
    '''
        :param  torch.Tensor    vec             : a tensor where each row consists of all channels of a vector.
        :param  List            affine_layers   : the affine layers of the synthesis network which defines the StyleSpace
        
        :return List                            : a list consists of all layers of style vectors
    '''
    idx = 0
    styles = []
    for layer, is_conv in affine_layers:
        layer_dim = layer.weight.shape[0]

        if len(vec.shape) == 1:
            styles.append(vec[idx:idx + layer_dim].clone().unsqueeze(0))
        elif len(vec.shape) == 2:
            styles.append(vec[:, idx:idx + layer_dim].clone())
        else:
            print("Invalid shape", vec.shape, ", input tensor 'edit' should be 1-dim of 2-dim.")
            exit(-1)
        idx += layer_dim
        # else:
        #     if len(vec.shape) == 1:
        #         styles.append(torch.zeros(layer_dim, device=vec.device, dtype=vec.dtype).unsqueeze(0))
        #     elif len(vec.shape) == 2:
        #         styles.append(torch.zeros((vec.shape[0], layer_dim), device=vec.device, dtype=vec.dtype))
        #     else:
        #         print("Invalid shape", vec.shape, ", input tensor 'edit' should be 1-dim of 2-dim.")
        #         exit(-1)

    return styles

def styles_to_vec(styles, affine_layers)->Tuple[torch.Tensor, np.ndarray]:
    '''
        :param  List            styles          : a list consists of layers of style vectors
        :param  List            affine_layers   : affine layers of the synthesis network

        :return torch.Tensor                    : the vector(s) where each row concatenates all layers (including toRGB layers) of a style vector
        :return np.ndarray                        : indexes of all channels in convolutional layers
    '''
    res_vec = None
    res_channels = []
    idx = 0
    ch_idx = 0
    for layer, is_conv in affine_layers:
        style = styles[idx]
        if res_vec is None:
            res_vec = style.clone()
        else:
            res_vec = torch.hstack((res_vec, style))
        if is_conv:
            res_channels += list(range(ch_idx, ch_idx + style.shape[-1]))
        idx += 1
        ch_idx += style.shape[-1]
    res_channels = np.array(res_channels)
    return res_vec, res_channels


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