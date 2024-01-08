import argparse
import os


def comma_separated_words(string):
    words = string.split(',')
    sorted_words = sorted(words)
    return sorted_words


parser = argparse.ArgumentParser(description="HyperSpectral Image Reconstruction Toolbox")

# Hardware specifications
parser.add_argument("--gpu_id", type=str, default='7')

# Data specifications
parser.add_argument('--path_root', default='/data3/zwz/', type=str,
                    help='root path')
parser.add_argument('--train_path_root', default='DUNTS/data/trainSet/', type=str,
                    help='root path of trainSet')
parser.add_argument('--valid_path_root', default='DUNTS/data/validSet/', type=str,
                    help='root path of testSet')
parser.add_argument('--test_path_root', default='DUNTS/data/testSet/', type=str,
                    help='root path of testSet')
parser.add_argument('--real_path_root', default='DUNTS/data/testReal/', type=str,
                    help='root path of testSet')
parser.add_argument('--result_path_root', default='DUNTS/data/result/', type=str,
                    help='root path of result')
parser.add_argument('--model_path_root', default='DUNTS/data/model/', type=str,
                    help='root path of result')
parser.add_argument('--folder_name', default='628', type=str,
                    help='path of test model')
parser.add_argument('--folder_name_test', default='628', type=str,
                    help='path of test data')

# Model specifications
parser.add_argument('--method', type=str, default='DUNTS', help='method name')

# Training specifications
parser.add_argument("--first_epoch", type=int, default=0, help='start epoch')
parser.add_argument("--max_epoch", type=int, default=5000, help='total epoch')
parser.add_argument("--test_epoch", type=int, default=800, help='test epoch')
parser.add_argument("--scheduler", type=str, default='MultiStepLR', help='MultiStepLR or CosineAnnealingLR')
parser.add_argument("--milestones", type=int, default=[400, 800, 1200, 1600, 2000], help='milestones for MultiStepLR')

# HyperParameter
parser.add_argument("--seed", default=1, type=int, help='Random_seed')
parser.add_argument("--size", default=64, type=int, help='cropped patch size')
parser.add_argument('--batch_size', type=int, default=16, help='the number of HSIs per batch')
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument('--channel', type=int, default=4, help='channel number')
parser.add_argument("--gamma", type=float, default=0.5, help='learning rate decay for MultiStepLR')
parser.add_argument('--rho', type=float, default=0.3, help='the weight of mid loss')
parser.add_argument('--stage', type=int, default=8, help='the stage number of iterative algorithms')
parser.add_argument('--loss', type=comma_separated_words, default='L1', help='the type of loss{L1,MSE,FFL}')
parser.add_argument('--bias_index', type=int, default=-1, help='bias index')
parser.add_argument('--sim', type=str, default=False, help='sim flag')

# Tensorboard
parser.add_argument('--tb', type=str, default=False, help='')
parser.add_argument('--flag', type=str, default='', help='')

opt = parser.parse_args()
opt.train_path_root = os.path.join(opt.path_root, opt.train_path_root)
opt.test_path_root = os.path.join(opt.path_root, opt.test_path_root)
opt.real_path_root = os.path.join(opt.path_root, opt.real_path_root)
opt.result_path_root = os.path.join(opt.path_root, opt.result_path_root)
opt.model_path_root = os.path.join(opt.path_root, opt.model_path_root)

for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False