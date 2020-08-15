import argparse
import tensorflow as tf
from model import OTB
import os

parser = argparse.ArgumentParser(description='Argument parser')

"""Main arguments"""
parser.add_argument('--mode', dest='mode', type=str, help='select ', choices=['reprojection', 'agreement', 'uniqueness', 'otb', 'otb-online'])
parser.add_argument('--checkpoint_path', dest='checkpoint_path', type=str, help='path to a specific checkpoint to load')
parser.add_argument('--output_path', dest='output_path', type=str, help='path where to save confidence maps')
parser.add_argument('--left_dir', dest='left_dir', type=str, help='path to left images')
parser.add_argument('--right_dir', dest='right_dir', type=str, help='path to right images')
parser.add_argument('--disp_dir', dest='disp_dir', type=str, help='path to disparity maps')

"""Optional arguments"""
parser.add_argument('--colors', help='Save color confidence maps', action='store_true')
parser.add_argument('--cpu', help='Run on cpu', action='store_true')
parser.add_argument('--mem', dest='mem', type=float, default=0.75, help='Portion of memory')
parser.add_argument('--initial_learning_rate', dest='initial_learning_rate', type=float, default=0.0001, help='initial learning rate for gradient descent')
parser.add_argument('--p', dest='p', type=str, nargs='+',default=['t','a','u'],choices=['t','a','u'])
parser.add_argument('--q', dest='q', type=str, nargs='+',default=['t'],choices=['t','a','u'])

"""Custom arguments for running on your own data"""
parser.add_argument('--image_height', dest='image_height', type=int, default=400, help='image height')
parser.add_argument('--image_width', dest='image_width', type=int, default=880, help='image width')
parser.add_argument('--dataset',  dest='dataset', type=str, default='filelist/drivingstereo.txt', help='dataset')

args = parser.parse_args()

def main(_):

    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    gpu_opt = tf.GPUOptions(per_process_gpu_memory_fraction=args.mem)

    assert len(args.p) != 0 or args.mode != 'otb-online', "--p cannot be empty!"
    assert len(args.q) != 0 or args.mode != 'otb-online', "--q cannot be empty!" 

    assert args.image_height % 16 == 0, "Image height must be multiple of 16"
    assert args.image_width % 16 == 0, "Image width must be multiple of 16"

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_opt)) as sess:
        model = OTB(sess,
                    mode=args.mode, dataset=args.dataset,
                    left_dir=args.left_dir, right_dir=args.right_dir, disp_dir=args.disp_dir,
                    image_height=args.image_height,
                    image_width=args.image_width,
                    p=args.p, q=args.q, colors=args.colors,
                    initial_learning_rate=args.initial_learning_rate)

        model.run(args)


if __name__ == '__main__':
    tf.app.run()
