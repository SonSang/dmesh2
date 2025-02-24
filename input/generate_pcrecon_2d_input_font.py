import os
import argparse
import numpy as np
import torch as th
from matplotlib import pyplot as plt

from input.common import DOMAIN
from exp.d2.renderer import render_point_cloud
from exp.d2.parser import BezierGlyph

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--font-path", type=str, default="data/2d/font/Roboto/Roboto-Regular.ttf")
    parser.add_argument("--font-char", type=str, default="A")
    parser.add_argument("--output-path", type=str, default="input/2d/pcrecon/")
    parser.add_argument("--num-sample", type=int, default=1000)
    parser.add_argument("--render", action='store_true')
    args = parser.parse_args()

    font_path = args.font_path
    font_char = args.font_char
    output_path = args.output_path
    num_sample = int(args.num_sample)
    render = args.render

    ### sample points
    bg = BezierGlyph(font_path)
    bg.load_character(font_char)
    pc = th.tensor(bg.sample(num_sample), dtype=th.float32)
    assert len(pc) > 0, "No points sampled."
    print("===== Number of points:", len(pc))

    pc = np.array(pc)
    
    ### normalize points
    pc_x_max = np.max(pc[:, 0])
    pc_x_min = np.min(pc[:, 0])
    pc_y_max = np.max(pc[:, 1])
    pc_y_min = np.min(pc[:, 1])

    pc_center_x = (pc_x_max + pc_x_min) * 0.5
    pc_center_y = (pc_y_max + pc_y_min) * 0.5
    pc_center = np.array([pc_center_x, pc_center_y])
    pc = pc - pc_center

    pc_size = np.linalg.norm(pc, ord=2, axis=-1).max()
    pc = (pc / pc_size) * DOMAIN

    ### save points
    basedir = os.path.dirname(output_path)
    os.makedirs(basedir, exist_ok=True)

    font_name = os.path.basename(font_path).split('.')[0]
    output_file_path = os.path.join(output_path, f'{font_name}_{font_char}.npy')

    np.save(output_file_path, pc)

    ### render
    if render:
        pc = th.tensor(pc).float()
        output_render_path = os.path.join(output_path, f'{font_name}_{font_char}.pdf')
        render_point_cloud(pc, domain=(-DOMAIN, DOMAIN), color='black', point_size=1e-3, fig_size=10, path=output_render_path)