from TacoTron import TacoTronTTS
from TransformerTTS import TransformerTTS
from tts_dataloader import symbols_len
from torch.optim import Adam
from torch import nn

tacotron_config = {
            'vocab_size': symbols_len,
            'embedding_dim' : 256,
            'enc_hidden_size': 128,
            'decoder_hidden_size' : 512,
            'pre_net_out_size' : 256,
            'location_n_filters' : 32,
            'location_kernel_size' : 31,
            'attn_dim' : 128,
            'post_net_channels' : 512,
            'post_net_kernel_size' : 5,
            'dropout_encoder' : 0.2,
            'dropout_decoder' : 0.1,
        }

transformer_config = {
            'vocab_size': symbols_len,
            'embedding_size': 256,
            'encoder_embedding_size': 512,
            'ff_hidden_size': 1024,
            'post_net_channels': 1024,
            'post_net_kernel_size': 5,
            'dropout_encoder': 0.2
        }

def get_model(model_name: str, mel_bins) -> nn.Module:
    model = None
    if model_name == 'Seq2SeqTTS_GRU':
        model =  TacoTronTTS(mel_bins=mel_bins, **tacotron_config)
    elif model_name == 'TransformerTTS':
        model =  TransformerTTS(mel_bins=mel_bins, **transformer_config)
    else:
        raise ValueError('Invalid model name for model hyper params.')
    return model

# return optimizer for specific model
def get_optimizer(model: nn.Module, lr=1e-4, weight_decay=1e-6):
    optimizer = None
    if type(model) is TacoTronTTS:
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif type(model) is TransformerTTS:
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError('Invalid model name for model hyper params.')
    return optimizer