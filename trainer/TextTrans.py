import time
import math
import torch
from tqdm import tqdm
from torch import nn, optim
from torch.optim import Adam

from dataset.datamanager import DataManeger
from model.transformer import Transformer
from trainer.tools import idx_to_word, get_bleu, epoch_time


class TextTranslator:
    """
    A standard task for transformer model: Natural Language Translator.
    """
    def __init__(self, args):
        """
        Function:
            step 1: initialize the configs for training text translator.
            step 2: load the dataset (multi30k).
            step 3: build transformer and optimizer.
        Args:
            args: configs include model parameters, training config, dataset and so on.
        """
        # GPU device setting
        self.device = torch.device(f"cuda:{args.gpu_id}" if args.gpu_id != -1 else 'cpu')

        # model parameter setting
        self.d_model = args.d_model
        self.n_layers = args.n_layers
        self.n_heads = args.n_heads
        self.d_ff = args.d_ff
        self.dropout = args.dropout
        self.max_len = args.max_len

        # optimizer parameter setting
        self.batch_size = args.batch_size
        self.init_lr = args.init_lr
        self.factor = args.factor
        self.adam_eps = args.adam_eps
        self.patience = args.patience
        self.warmup = args.warmup
        self.epochs = args.epochs
        self.clip = args.clip
        self.weight_decay = args.weight_decay
        self.inf = float('inf')

        # load data and build model 
        self.load_data(args.src_mode, args.tgt_mode, args.data_path)
        self.build_model()

    def load_data(self, src_mode, tgt_mode, data_path):
        print(">>> Loading the dataset...")
        self.data_manager = DataManeger(src_mode, tgt_mode, data_path)
        train, valid, test = self.data_manager.make_dataset()

        self.train_iter, self.valid_iter, self.test_iter = self.data_manager.make_iter(train, valid, test,
                                                            batch_size=self.batch_size)
        
        self.src_pad_idx = self.data_manager.src_vocab.get_stoi()['<pad>']
        self.tgt_pad_idx = self.data_manager.tgt_vocab.get_stoi()['<pad>']
        self.tgt_bos_idx = self.data_manager.tgt_vocab.get_stoi()['<bos>']
        self.enc_voc_size = len(self.data_manager.src_vocab)
        self.dec_voc_size = len(self.data_manager.tgt_vocab)
        print(">>> Loading the dataset... Done! <<<")

    def build_model(self):
        print(">>> Building the transformer model...")
        self.model = Transformer(src_pad_idx=self.src_pad_idx,
                tgt_pad_idx=self.tgt_pad_idx,
                tgt_bos_idx=self.tgt_bos_idx,
                enc_voc_size=self.enc_voc_size,
                dec_voc_size=self.dec_voc_size,
                d_model=self.d_model,
                n_head=self.n_heads,
                max_len=self.max_len,
                d_ff=self.d_ff,
                n_layers=self.n_layers,
                dropout=self.dropout,
                device=self.device).to(self.device)

        params_num, params_size = self.count_parameters_and_size(self.model)
        print(f'>>> trainable parameters: {params_num} trainable parameters size: {params_size} MB <<<')

        self.model.apply(self.initialize_weights)
        print(">>> Building the transformer model... Done! <<<")

        print(">>> Building the optimizer, scheduler and criterion...")
        self.optimizer = Adam(params=self.model.parameters(),
                        lr=self.init_lr,
                        weight_decay=self.weight_decay,
                        eps=self.adam_eps)
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
                                                verbose=True,
                                                factor=self.factor,
                                                patience=self.patience)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.src_pad_idx)
        print(">>> Building the optimizer, scheduler and criterion... Done! <<<")
    
    def train(self):
        best_loss = self.inf
        train_losses, test_losses, bleus = [], [], []
        for step in range(self.epochs):
            start_time = time.time()
            train_loss = self.train_epoch()
            valid_loss, bleu = self.evaluate()
            end_time = time.time()

            if step > self.warmup:
                self.scheduler.step(valid_loss)

            train_losses.append(train_loss)
            test_losses.append(valid_loss)
            bleus.append(bleu)
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(self.model.state_dict(), 'checkpoints/best-model.pt')

            f = open('output/train_loss.txt', 'w')
            f.write(str(train_losses))
            f.close()

            f = open('output/bleu.txt', 'w')
            f.write(str(bleus))
            f.close()

            f = open('output/test_loss.txt', 'w')
            f.write(str(test_losses))
            f.close()

            print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'Val Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
            print(f'BLEU Score: {bleu:.3f}')

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0

        start_time = time.time()
        for i, batch in enumerate(self.train_iter):
            batch_start_time = time.time()

            src, tgt = batch[0].to(self.device), batch[1].to(self.device)

            self.optimizer.zero_grad()
            output = self.model(src, tgt[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            tgt = tgt[:, 1:].contiguous().view(-1)

            loss = self.criterion(output_reshape, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()

            epoch_loss += loss.item()

            batch_time = time.time() - batch_start_time
            elapsed_time = time.time() - start_time
            remaining_time = (len(self.train_iter) - i - 1) * batch_time

            remaining_hours, rem = divmod(remaining_time, 3600)
            remaining_minutes, remaining_seconds = divmod(rem, 60)
            print(f'[Step: {i + 1}/{len(self.train_iter)} | '
                f'Batch Loss: {loss.item():.4f} | '
                f'Batch Time: {batch_time:.2f}s | '
                f'Elapsed Time: {elapsed_time:.2f}s | '
                f'Estimated Remaining Time: {int(remaining_hours)}h '
                f'{int(remaining_minutes)}m {int(remaining_seconds)}s]')

        return epoch_loss / len(self.train_iter)

    def evaluate(self):
        self.model.eval()
        epoch_loss = 0
        batch_bleu = []
        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.valid_iter)):
                src, tgt = batch[0].to(self.device), batch[1].to(self.device)

                output = self.model(src, tgt[:, :-1])
                output_reshape = output.contiguous().view(-1, output.shape[-1])
                tgt = tgt[:, 1:].contiguous().view(-1)

                loss = self.criterion(output_reshape, tgt)
                epoch_loss += loss.item()

                total_bleu = []
                for j in range(self.batch_size):
                    try:
                        tgt_words = idx_to_word(batch[1][j], self.data_manager.tgt_vocab)
                        output_words = output[j].max(dim=1)[1]
                        output_words = idx_to_word(output_words, self.data_manager.tgt_vocab)
                        bleu = get_bleu(hypotheses=output_words.split(), reference=tgt_words.split())
                        total_bleu.append(bleu)
                    except:
                        pass

                total_bleu = sum(total_bleu) / len(total_bleu)
                batch_bleu.append(total_bleu)

        batch_bleu = sum(batch_bleu) / len(batch_bleu)
        return epoch_loss / len(self.valid_iter), batch_bleu

    def count_parameters_and_size(self, model):
        """
        Count the learning parameters and size in constructed transformer.
        """
        param_size_in_bytes = 4    # default float32
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        total_size_in_MB = total_params * param_size_in_bytes / (1024 ** 2)
        return total_params, total_size_in_MB

    def initialize_weights(self, m):
        """
        Initialize the weights using Kaiming He method in constructed transformer.
        """
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.kaiming_uniform(m.weight.data)
    
        
            