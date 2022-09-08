import click
import numpy as np
import torch
import pickle
import clip
from PIL import Image
import tqdm
from utils.edit_utils import get_affine_layers, to_styles, vec_to_styles, w_to_styles, styles_to_vec
from utils.image_utils import tensor2pil
from utils.model_utils import load_g, load_from_pkl_model

DEVICE = 'cuda:0'
ALPHA = 5
BATCH_SIZE = 4

def get_w(z, mapping):
    return mapping.forward(z, c=18)


from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def simple_preprocess(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def get_image_features(styles_vec, synthesis, affine_layers, clip_model, clip_preprocess):
    image_features = None
    batch_num = int(np.ceil(styles_vec.shape[0] / BATCH_SIZE))

    # for i in range(styles_vec.shape[0]):
    for bi in range(batch_num):
        batch_start = bi * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, styles_vec.shape[0])

        batch_style = vec_to_styles(styles_vec[batch_start:batch_end], affine_layers)
        tensor = synthesis.forward(batch_style, style_input=True, noise_mode='const', force_fp32=True).detach()

        # pil = tensor2pil(tensor)
        # image = clip_preprocess(pil).detach().unsqueeze(0).to(DEVICE)
        batch_image = clip_preprocess(tensor).detach().to(DEVICE)
        batch_image_feature = clip_model.encode_image(batch_image).detach()

        if image_features is None:
            image_features = batch_image_feature
        else:
            image_features = torch.cat((image_features, batch_image_feature), dim=0)

    return image_features


@click.command()
@click.option('--model_path', type=str, required=True)
@click.option('--output_path', type=str, required=True)
def _main(model_path, output_path):
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

    preprocess = simple_preprocess(model.visual.input_resolution)
    model = model.eval().requires_grad_(False)

    # load random data
    print("Loading random data...")
    rand_z = np.load('data/random_z.npy')
    rand_z = torch.Tensor(rand_z).to(DEVICE)
    rand_w = get_w(rand_z, mapping)
    rand_styles = w_to_styles(rand_w, affine_layers)

    rand_styles_vec = styles_to_vec(rand_styles, affine_layers)
    std = torch.std(rand_styles_vec, dim=0, unbiased=True)

    print("Calculating original image features...")
    i0 = get_image_features(rand_styles_vec, synthesis, affine_layers, model, preprocess)
    
    print("Calculating di for each channel...")

    di_all = None
    for ci in tqdm.tqdm(range(rand_styles_vec.shape[1])):
        styles_vec_delta = rand_styles_vec.clone()
        styles_vec_delta[:, ci] += ALPHA * std[ci]

        i1 = get_image_features(styles_vec_delta, synthesis, affine_layers, model, preprocess)
        di = i1 - i0
        di = torch.mean(di, dim=0)
        di = di / torch.norm(di)
        if di_all is None:
            di_all = di
        else:
            di_all = torch.vstack((di_all, di))
    print(di_all.shape)
    di_all = di_all.cpu().numpy()
    np.save(output_path, di_all)


if __name__ == "__main__":
    _main()