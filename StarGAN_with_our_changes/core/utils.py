import numpy as np
import torch
import torchvision.utils as vutils


def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def save_image(x, filename):
    x = denormalize(x)
    vutils.save_image(x.cpu(), filename, padding=0)


@torch.no_grad()
def translate_using_reference(nets, args, x_src, y_src, x_ref, y_ref, filename):
    N, C, H, W = x_src.size()
    for j, (x_s, y_s) in enumerate(zip(x_src, y_src)):
        x_s = torch.unsqueeze(x_s, dim=0)
        for i, (x_r, y_r) in enumerate(zip(x_ref, y_ref)):
            x_r = torch.unsqueeze(x_r, dim=0)
            y_r = torch.unsqueeze(y_r, dim=0)
            masks = nets.fan.get_heatmap(x_s) if args.w_hpf > 0 else None
            s_ref = nets.style_encoder(x_r, y_r)
            x_fake = nets.generator(x_s, s_ref, masks=masks)
            save_image(x_fake, f'{filename}/{y_s}/new_image_{y_s}_{i}_{j}.jpg')
