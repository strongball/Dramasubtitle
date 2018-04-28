#from translate.train import trainer
#from SubToSub.train import trainer
from ImgSub2Sub.train import trainer
#from NGram.train import trainer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epoch', help="Epoch to Train", type=int, default=10)
parser.add_argument('-b', '--batch', help="Batch size", type=int, default=64)
parser.add_argument('-lr', help="learning rate to Train", type=float, default = 1e-4)
parser.add_argument('-m', '--model', help="Model dir", required=True)
parser.add_argument('-d', '--data', help="Data dir", required=True)
parser.add_argument('-c', '--checkpoint', help="Old Model to load", default = "NoneExist")

if __name__ == "__main__":
    args = parser.parse_args()
    trainer(args)