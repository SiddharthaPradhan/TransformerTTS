import torch
from torch.utils.data import DataLoader, Dataset
from datasets import Dataset as HfDataset
from torchaudio.transforms import MelSpectrogram
from typing import Tuple
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from utils import SpeechConverter

EOS = 'EOS'
PAD = 'PAD'

symbols = [
    PAD, EOS, ' ', '!', ',', '-', '.', \
    ';', '?', 'a', 'b', 'c', 'd', 'e', 'f', \
    'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', \
    'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'à', \
    'â', 'è', 'é', 'ê', 'ü', '’', '“', '”' \
]

symbols_len = len(symbols)

symbol_to_idx = {
  s: i for i, s in enumerate(symbols)
}

idx_to_symbol = {v: k for k, v in symbol_to_idx.items()}

def seq_to_text(seq, remove_pad = True):
    text = ""
    for idx in seq:
        symbol = idx_to_symbol.get(idx.item(), None)
        if remove_pad and symbol == PAD:
            symbol = ""
        text += symbol
    return text

def text_to_seq_char_level(text):
    text = text.lower()
    seq = []
    for symbol in text:
        idx = symbol_to_idx.get(symbol, None)
        if idx is not None:
            seq.append(idx)
    seq.append(symbol_to_idx.get(EOS))
    return torch.IntTensor(seq)

class LjSpeechDataset(Dataset):
    def __init__(self, hf_dataset: HfDataset, num_mels=80,
                 text_col='normalized_text', text_to_seq_fn=text_to_seq_char_level):
        self.hf_dataset = hf_dataset
        self.num_mels = num_mels
        self.text_col = text_col
        self.text_to_seq_fn = text_to_seq_fn
        self.speech_converter = SpeechConverter(self.num_mels)

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx) -> Tuple[torch.IntTensor, torch.Tensor]:
        text = self.hf_dataset[idx][self.text_col]
        audio_waveform = self.hf_dataset[idx]['audio']['array']
        # sr is constant for the dataset, use speech_utils.sr
        # sampling_rate = self.hf_dataset[idx]['audio']['sampling_rate']
        
        # Apply text_to_seq_fn to the text
        text_seq = self.text_to_seq_fn(text)
        mel_spec = self.speech_converter.convert_to_mel_spec(audio_waveform)
        
        return text_seq, mel_spec

def speech_collate_fn(batch):
    # sort the batch based on input text (this is needed for pack_padded_sequence)
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    text_seqs, mel_specs = zip(*batch)
    text_seq_lens = [text_seq.shape[-1] for text_seq in text_seqs] # batch first
   
    mel_specs_t = []
    mel_spec_lens = []
    max_mel_seq = -1
    for mel_spec in mel_specs:
        mel_specs_t.append(mel_spec.T)
        true_mel_size = mel_spec.shape[-1]
        mel_spec_lens.append(true_mel_size)
        if true_mel_size > max_mel_seq:
            max_mel_seq = true_mel_size
    # need to know max size to pad to/ generate stop token input
    stop_token_targets = []
    for i in range(len(mel_specs)):
        stop_token_target = torch.zeros(max_mel_seq)
        true_mel_size = mel_spec_lens[i]
        stop_token_target[true_mel_size-1:] = 1
        stop_token_targets.append(stop_token_target)
    
    # pad sequence so pytorch can batch them together
    padded_text_seqs = pad_sequence(text_seqs, batch_first=True, padding_value=symbol_to_idx.get(PAD))
    padded_mel_specs = pad_sequence(mel_specs_t, batch_first=True, padding_value=0)
    text_seq_lens = torch.IntTensor(text_seq_lens)
    mel_spec_lens = torch.IntTensor(mel_spec_lens)
    stop_token_targets = torch.stack(stop_token_targets)
    # print("In collate", padded_mel_specs.shape, stop_token_targets.shape)
    return padded_text_seqs, text_seq_lens, padded_mel_specs, mel_spec_lens, stop_token_targets

def get_data_loader(dataset: HfDataset, batch_size, shuffle=True, num_workers=0) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=speech_collate_fn, 
                      num_workers=num_workers, pin_memory=True)

def load_data(batch_size, mel_bins=128, subsample_ratio=None):
    # load dataset
    hf_dataset = load_dataset('keithito/lj_speech')['train']
    if subsample_ratio is not None: # Used for testing model arch
        hf_dataset = hf_dataset.train_test_split(train_size=subsample_ratio)['train']
    hf_dataset.set_format(type="torch", columns=["audio"], output_all_columns=True)
    # split dataset into training and (validation+test) set
    hf_split_datadict = hf_dataset.train_test_split(test_size=0.2)
    hf_train_dataset = hf_split_datadict['train']
    # split (validation+test) dataset into validation and test set
    hf_dataset = hf_split_datadict['test']
    hf_split_datadict = hf_dataset.train_test_split(test_size=0.5)
    hf_val_dataset = hf_split_datadict['train']
    hf_test_dataset = hf_split_datadict['test']
    print(f'Dataset Sizes: Train ({len(hf_train_dataset)}), Val ({len(hf_val_dataset)}), Test ({len(hf_test_dataset)})')
    # convert hf_dataset to pytorch datasets
    train_ds = LjSpeechDataset(hf_train_dataset, num_mels=mel_bins)
    val_ds = LjSpeechDataset(hf_val_dataset, num_mels=mel_bins)
    test_ds = LjSpeechDataset(hf_test_dataset, num_mels=mel_bins)
    # convert datasets to dataloader
    train_dl = get_data_loader(train_ds, batch_size, num_workers=3)
    val_dl = get_data_loader(val_ds, batch_size, shuffle=False, num_workers=1)
    test_dl = get_data_loader(test_ds, batch_size, shuffle=False, num_workers=1)
    return train_dl, val_dl, test_dl