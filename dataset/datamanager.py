import io
from collections import Counter
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
torchtext.disable_torchtext_deprecation_warning()
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab
from torchtext.utils import extract_archive


class DataManeger:
    """
    A integrated data manager with builded tokenizer and vocabulary.
    """
    def __init__(self, src_mode, tgt_mode, data_path):
        """
        Args:
            src_mode: source natural language, ('en': English, 'de': Deutsch / German', 'cs': Čeština / Czech, 'fr': Français / French).
            tgt_mode: target natural language, ('en': English, 'de': Deutsch / German', 'cs': Čeština / Czech, 'fr': Français / French).
            data_path: the path of dataset.
        """
        self.src_mode = src_mode
        self.tgt_mode = tgt_mode

        self.tokenize_src = get_tokenizer('spacy', language=src_mode)
        self.tokenize_tgt = get_tokenizer('spacy', language=tgt_mode)

        train_urls = ('train.'+ src_mode +'.gz', 'train.'+ tgt_mode +'.gz')
        val_urls = ('val.'+ src_mode +'.gz', 'val.'+ tgt_mode +'.gz')
        test_urls = ('test_2016_flickr.'+ src_mode +'.gz', 'test_2016_flickr.'+ tgt_mode +'.gz')

        self.train_filepaths = [extract_archive(data_path + url)[0] for url in train_urls]
        self.val_filepaths = [extract_archive(data_path + url)[0] for url in val_urls]
        self.test_filepaths = [extract_archive(data_path + url)[0] for url in test_urls]

        self.src_vocab = self.build_vocab(self.tokenize_src, self.train_filepaths[0])
        self.tgt_vocab = self.build_vocab(self.tokenize_tgt, self.train_filepaths[1])

        self.src_vocab.set_default_index(self.src_vocab['<unk>'])
        self.tgt_vocab.set_default_index(self.tgt_vocab['<unk>'])


    def make_dataset(self):
        """
        Process out the data through their zip files.
        """
        train_data = self.data_process(self.train_filepaths)
        val_data = self.data_process(self.val_filepaths)
        test_data = self.data_process(self.test_filepaths)

        return train_data, val_data, test_data

    def build_vocab(self, tokenizer, train_filepath):
        """
        Build the corresponding vocabulary for the two languages.
        """
        counter = Counter()
        with io.open(train_filepath, encoding="utf8") as f:
            for string_ in f:
                counter.update(tokenizer(string_))
        return vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

    def data_process(self, filepaths):
        """
        Create the input_id tensors using tokenizer and vocabulary.
        """
        raw_src_iter = iter(io.open(filepaths[0], encoding="utf8"))
        raw_tgt_iter = iter(io.open(filepaths[1], encoding="utf8"))
        data = []
        for (raw_src, raw_tgt) in zip(raw_src_iter, raw_tgt_iter):
            src_tensor = torch.tensor([self.src_vocab[token] for token in self.tokenize_src(raw_src)],
                                    dtype=torch.long)
            tgt_tensor = torch.tensor([self.tgt_vocab[token] for token in self.tokenize_tgt(raw_tgt)],
                                    dtype=torch.long)
            data.append((src_tensor, tgt_tensor))
        return data

    def make_iter(self, train, validate, test, batch_size):
        """
        Create the iterater for sub-dataset using collection function.
        """
        train_iter = DataLoader(train, batch_size=batch_size,
                                shuffle=True, collate_fn=self.generate_batch)
        valid_iter = DataLoader(validate, batch_size=batch_size,
                                shuffle=False, collate_fn=self.generate_batch)
        test_iter = DataLoader(test, batch_size=batch_size,
                            shuffle=False, collate_fn=self.generate_batch)
        return train_iter, valid_iter, test_iter

    def generate_batch(self, data_batch):
        """
        Construct the batch input_id tensors, add the bos and eos tokens and padding the sentence.
        """
        SRC_PAD_IDX, TGT_PAD_IDX = self.src_vocab['<pad>'], self.tgt_vocab['<pad>']
        SRC_BOS_IDX, TGT_BOS_IDX = self.src_vocab['<bos>'], self.tgt_vocab['<bos>']
        SRC_EOS_IDX, TGT_EOS_IDX = self.src_vocab['<eos>'], self.tgt_vocab['<eos>']
        src_batch, tgt_batch = [], []
        for (src_item, tgt_item) in data_batch:
            src_batch.append(torch.cat([torch.tensor([SRC_BOS_IDX]), src_item, torch.tensor([SRC_EOS_IDX])], dim=0))
            tgt_batch.append(torch.cat([torch.tensor([TGT_BOS_IDX]), tgt_item, torch.tensor([TGT_EOS_IDX])], dim=0))

        # padding the sentence using PAD_IDX
        src_batch = pad_sequence(src_batch, padding_value=SRC_PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=TGT_PAD_IDX)
        return src_batch.t(), tgt_batch.t()


