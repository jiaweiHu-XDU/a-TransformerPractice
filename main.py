import warnings
import argparse
from trainer.TextTrans import TextTranslator

warnings.filterwarnings('ignore')

def create_args_parser():
    parser = argparse.ArgumentParser(description="Train text translator based on a standard transformer architecture.")
    
    # model parameter setting
    parser.add_argument('--d_model', type=int, default=512, help="dimension for a token")
    parser.add_argument('--n_layers', type=int, default=6, help="number of encoder and decoder layers")
    parser.add_argument('--n_heads', type=int, default=8, help="number of attention heads")
    parser.add_argument('--d_ff', type=int, default=2048, help="dimension of hidden layer in feed forward network")
    parser.add_argument('--max_len', type=int, default=256, help="max length of tokens for a single sentence")
    parser.add_argument('--dropout', type=float, default=0.1, help="dropout probability")

    # optimizer parameter setting
    parser.add_argument('--batch_size', type=int, default=128, help="batch size for training")
    parser.add_argument('--epochs', type=int, default=10, help="number of training epochs")
    parser.add_argument('--warmup', type=int, default=100, help="number of warmup steps for learning rate")
    parser.add_argument('--init_lr', type=float, default=1e-5, help="init learning rate for optimizer")
    parser.add_argument('--factor', type=float, default=0.9, help="learning rate decay factor for optimizer")
    parser.add_argument('--adam_eps', type=float, default=5e-9, help="epsilon value for the Adam optimizer")
    parser.add_argument('--patience', type=int, default=10, help="number of epochs with no improvement in validation loss")
    parser.add_argument('--clip', type=float, default=1.0, help="gradient clipping value")
    parser.add_argument('--weight_decay', type=float, default=5e-4, help="weight decay (L2 regularization) factor")

    # device and dataset setting
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU device ID to use (if 'gpu_id' >= 0). 'gpu_id' is -1 it will use CPU.")
    parser.add_argument('--src_mode', type=str, default='en', help="source language, to encode")
    parser.add_argument('--tgt_mode', type=str, default='de', help="target language, to decode")
    parser.add_argument('--data_path', type=str, default='./data/multi30k/task1/raw/', help="path of training dataset")
    
    return parser

def print_args(args):
    print(">>> Training text translator configs...")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

if __name__ == "__main__":
    parser = create_args_parser()
    args = parser.parse_args()

    print_args(args)

    translator = TextTranslator(args)
    # training the translator
    translator.train()