from Trainer import Trainer, TTS_Loss
import tts_dataloader
import torch
import numpy as np
import os
from models import get_model, get_optimizer

class Pipeline:
    def __init__(self, model_name, mel_bins, batch_size, lr, max_epochs, weight_decay, checkpoint_prefix, 
                subsample_ratio=None, teacher_f_ratio=0.5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        self.model = get_model(model_name, mel_bins).to(device)
        optimizer = get_optimizer(self.model, lr, weight_decay) 
        train_dl, val_dl, test_dl = tts_dataloader.load_data(batch_size, mel_bins=mel_bins, subsample_ratio=subsample_ratio)
        criterion = TTS_Loss().to(device)
        self.trainer = Trainer(mel_bins, self.model, max_epochs, optimizer, criterion,
                      train_dl, val_dl, test_dl, device, checkpoint_prefix, teacher_f_ratio=teacher_f_ratio, 
                      grad_clip=True, max_norm=1.0)

    def model_info(self):
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        num_params = sum([np.prod(p.size()) for p in params])
        print(self.model)
        print(f'Model has {num_params} trainable parameters')
        
    def run(self):
        self.model_info()
        self.trainer.train()
        # self.trainer.evaluate_on_test()


# Seq2SeqTTS = 'Seq2SeqTTS_GRU'
TransformerTTS = 'TransformerTTS'

if __name__ == "__main__":
    model_name = TransformerTTS  # Model name
    # Checkpoint location
    checkpoint_dir = f'checkpoints/{model_name}'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_prefix = checkpoint_dir
    # hyperparams
    mel_bins = 80
    batch_size = 128
    lr = 3*1e-4
    weight_decay = 1e-6
    max_epochs = 10000
    subsample_ratio = None # For testing arch (value of None for actual training)

    # setup training pipeline
    Pipeline(model_name, mel_bins, batch_size, lr, max_epochs, weight_decay, checkpoint_prefix,
             subsample_ratio=subsample_ratio, teacher_f_ratio=1).run()