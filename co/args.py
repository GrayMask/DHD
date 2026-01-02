import argparse
import os
from .utils import str2bool


def parse_args():
    parser = argparse.ArgumentParser()
    #
    parser.add_argument('--output_dir',
                        help='Output directory',
                        default='./output', type=str)
    parser.add_argument('--loss',
                        default='dispref', choices=['disp','ref','dispref'], type=str)
    parser.add_argument('--data_type',
                        default='syn', choices=['syn','real'], type=str)
    parser.add_argument('--channel',
                        help='Training channels',
                        default=3, choices=[1,3], type=int)
    parser.add_argument('--wavelength',
                        help='Wavelength',
                        default=27, type=int)
    #
    parser.add_argument('--cmd', 
                        help='Start training or test', 
                        default='resume', choices=['retrain', 'resume', 'retest', 'test_init'], type=str)
    parser.add_argument('--epoch', 
                        help='If larger than -1, retest on the specified epoch',
                        default=-1, type=int)
    parser.add_argument('--epochs',
                        help='Training epochs',
                        default=200, type=int)

    # 
    parser.add_argument('--ms',
                        help='If true, use multiscale loss',
                        default=True, type=str2bool)
    parser.add_argument('--pattern_path',
                        help='Path of the pattern image',
                        default='./data/colorpattern_camill.png', type=str)
    #
    parser.add_argument('--d_weight',
                        help='Weight of the disparity edge loss',
                        default=1, type=float)
    parser.add_argument('--de_weight',
                        help='Weight of the disparity edge loss',
                        default=100, type=float)
    parser.add_argument('--pat_weight',
                        help='Weight of the pattern loss',
                        default=0.2, type=float)
    parser.add_argument('--r_weight',
                        help='Weight of the reflectance loss',
                        default=1, type=float)
    parser.add_argument('--re_weight',
                        help='Weight of the reflectance edge loss',
                        default=8, type=float)
    #
    parser.add_argument('--lcn_radius',
                        help='Radius of the window for LCN pre-processing',
                        default=5, type=int)
    parser.add_argument('--max_disp',
                        help='Maximum disparity',
                        default=128, type=int)
    #
    parser.add_argument('--track_length',
                        help='Track length for geometric loss',
                        default=2, type=int)
    #
    parser.add_argument('--blend_im',
                        help='Parameter for adding texture',
                        default=0.6, type=float)

    parser.add_argument('--num_workers',
                        help='num_workers of dataloader',
                        default=20, type=int)

    parser.add_argument('--useAllGPU',
                        help='use all GPU to start',
                        default=False, action='store_true')
    
    args = parser.parse_args()

    args.exp_name = get_exp_name(args)

    return args


def get_exp_name(args):
    name = f"exp_{args.data_type}"
    return name



