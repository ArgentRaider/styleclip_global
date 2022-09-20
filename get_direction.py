from email.policy import default
import numpy as np
from PIL import Image
import clip
import torch

from utils.edit_utils import get_affine_layers, to_styles, vec_to_styles, w_to_styles, styles_to_vec
from utils.model_utils import load_g, load_from_pkl_model
from utils.image_utils import tensor2pil
from utils.styleclip_global_utils import get_direction
import click

def concat_images_horizontally(*images: Image.Image):
    assert all(image.height == images[0].height for image in images)
    total_width = sum(image.width for image in images)

    new_im = Image.new(images[0].mode, (total_width, images[0].height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.width

    return new_im

DEVICE = 'cuda:0'

@click.command()
@click.option('--model_path', type=str, required=True)
@click.option('--di_path', type=str, required=True)
@click.option('--diretion_name', type=str, required=True)
@click.option('--beta', type=float, required=True)
@click.option('--factor', type=float, default=2)

def _main(model_path, di_path, direction_name, beta, factor):
    # import network
    print("Importing", model_path, "...")
    G = load_g(model_path, DEVICE)
    G = load_from_pkl_model(G)
    mapping = G.mapping
    synthesis = G.synthesis
    affine_layers = get_affine_layers(synthesis)

    # import clip
    print("Importing CLIP...")
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)

    # import data
    print("Importing w+ data")
    w = np.load('data/neutral_pivots.npy')
    w = torch.Tensor(w).to(DEVICE)
    styles = w_to_styles(w, affine_layers)

    # original image
    orig_img = synthesis.forward(w, style_input=False, noise_mode='const', force_fp32=True)
    orig_img = tensor2pil(orig_img)
    orig_img.save('results/orig.png')

    di = np.load(di_path)
    edit_directions = get_direction('face', direction_name, beta, model=model, di=di)
    edit = to_styles(edit_directions, affine_layers)

    pos_edited_styles = [style + factor * edit_direction for style, edit_direction in zip(styles, edit)]
    neg_edited_styles = [style - factor * edit_direction for style, edit_direction in zip(styles, edit)]

    orig_tensor = synthesis.forward(styles, style_input=True, noise_mode='const',
                                                  force_fp32=True)
    pos_edited_tensor = synthesis.forward(pos_edited_styles, style_input=True, noise_mode='const',
                                                  force_fp32=True)
    neg_edited_tensor = synthesis.forward(neg_edited_styles, style_input=True, noise_mode='const',
                                                  force_fp32=True)
    orig_img = tensor2pil(orig_tensor)
    pos_edited_img = tensor2pil(pos_edited_tensor)
    neg_edited_img = tensor2pil(neg_edited_tensor)
    display_img = concat_images_horizontally(orig_img, pos_edited_img, neg_edited_img)
    display_img = display_img.resize((display_img.width // 2, display_img.height // 2))


if __name__ == "__main__":
    _main()