import torch
import torch.nn as nn
import torchaudio.transforms as T
import math
from tqdm import tqdm

def get_mask_from_lens(lens, max_len):
    base_mask = torch.arange(max_len).expand(lens.size(0), max_len).T.to(lens.device)

    mask = (base_mask > lens).permute(1,0)
    return mask


class EncoderPreNet(nn.Module):
    def __init__(self, vocab_size, encoder_embedding_size, embedding_size, dropout=0.0):
        super(EncoderPreNet, self).__init__()
        self.enc_embedding = nn.Embedding(vocab_size, encoder_embedding_size)
        self.linear_1 = nn.Linear(encoder_embedding_size, encoder_embedding_size)

        self.enc_conv1 = nn.Conv1d(encoder_embedding_size, encoder_embedding_size, kernel_size=3, padding=1, stride=1, dilation=1)
        self.enc_batch_norm_1 = nn.BatchNorm1d(encoder_embedding_size)
        self.enc_conv2 = nn.Conv1d(encoder_embedding_size, encoder_embedding_size, kernel_size=3, padding=1, stride=1, dilation=1)
        self.enc_batch_norm_2 = nn.BatchNorm1d(encoder_embedding_size)
        self.enc_conv3 = nn.Conv1d(encoder_embedding_size, encoder_embedding_size, kernel_size=3, padding=1, stride=1, dilation=1)
        self.enc_batch_norm_3 = nn.BatchNorm1d(encoder_embedding_size)
        self.dropout = nn.Dropout(dropout)

        

        self.linear_2 = nn.Linear(encoder_embedding_size, embedding_size)

    def forward(self, x: torch.Tensor):
        x = self.enc_embedding(x) # batch_size, max_seq_len, embedding_dim
        x = self.linear_1(x)
        x = x.permute(0, 2, 1)
        # First Conv Layer
        x = self.enc_conv1(x)
        x = self.enc_batch_norm_1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        # Second Conv Layer
        x = self.enc_conv2(x)
        x = self.enc_batch_norm_2(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        # Final conv layer
        x = self.enc_conv3(x)
        x = self.enc_batch_norm_3(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x = self.linear_2(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, embedding_size, ff_hidden_size):
        self.embedding_size = embedding_size
        self.ff_hidden_size = ff_hidden_size
        super(EncoderBlock, self).__init__()
        self.norm_1 = nn.LayerNorm(
            normalized_shape=self.embedding_size
        )

        self.attn = torch.nn.MultiheadAttention(
            embed_dim=self.embedding_size,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        self.norm_2 = nn.LayerNorm(
            normalized_shape = self.embedding_size
        )

        self.linear_1 = nn.Linear(
            embedding_size,
            ff_hidden_size
        )

        self.linear_2 = nn.Linear(
            ff_hidden_size,
            embedding_size
        )
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        x_residual, _ = self.attn(query=x, key=x, value=x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x_residual = self.dropout(x_residual)
        x = x + x_residual # add residual 1
        x = self.norm_1(x)
        x_residual = self.linear_1(x)
        x_residual = nn.functional.relu(x_residual)
        x_residual = self.dropout(x_residual)
        x_residual = self.linear_2(x_residual)
        x_residual = self.dropout(x_residual)
        x = x + x_residual # add residual 2
        x = self.norm_2(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, embedding_size, ff_hidden_size):
        self.embedding_size = embedding_size
        self.ff_hidden_size = ff_hidden_size
        super(DecoderBlock, self).__init__()
        self.norm_1 = nn.LayerNorm(self.embedding_size)
        self.self_attn = torch.nn.MultiheadAttention(embed_dim=self.embedding_size, num_heads=4, 
                                                     dropout=0.1, batch_first=True)
        self.dropout_1 = torch.nn.Dropout(0.1)
        self.norm_2 = nn.LayerNorm(normalized_shape=self.embedding_size)
        self.attn = torch.nn.MultiheadAttention(embed_dim=self.embedding_size, num_heads=4,
                                                dropout=0.1, batch_first=True)
        self.dropout_2 = torch.nn.Dropout(0.1)
        self.norm_3 = nn.LayerNorm(normalized_shape=self.embedding_size)
        self.linear_1 = nn.Linear(self.embedding_size, self.ff_hidden_size)
        self.dropout_3 = torch.nn.Dropout(0.1)
        self.linear_2 = nn.Linear(self.ff_hidden_size, self.embedding_size)
        self.dropout_4 = torch.nn.Dropout(0.1)

    def forward(self, x, memory, mel_self_mask=None, mel_pad_mask=None, look_ahead_mask=None, text_pad_mask=None):
        x_residual, _ = self.self_attn(query=x, key=x, value=x, attn_mask=mel_self_mask, 
                                       key_padding_mask=mel_pad_mask, need_weights=False)
        x_residual = self.dropout_1(x_residual)
        x = self.norm_1(x + x_residual)

        x_residual, attn_weights = self.attn(query=x, key=memory, value=memory, attn_mask=look_ahead_mask,
                             key_padding_mask=text_pad_mask, average_attn_weights=True)
        x_residual = self.dropout_2(x_residual)
        x = self.norm_2(x + x_residual)

        x_residual = self.linear_1(x)
        x_residual = nn.functional.relu(x_residual)
        x_residual = self.dropout_3(x_residual)
        x_residual = self.linear_2(x_residual)
        x_residual = self.dropout_4(x_residual)
        x = self.norm_3(x + x_residual)
        return x, attn_weights


# Pseudo-embedding layer for mel frame 
class DecoderPreNet(nn.Module):
    def __init__(self, mel_bins=128, out_size=128):
        super(DecoderPreNet, self).__init__()
        self.input_proj = nn.Linear(mel_bins, out_size, bias=False)
        self.out_proj = nn.Linear(out_size, out_size, bias=False)

    def forward(self, mel_frames: torch.Tensor) -> torch.Tensor:
        out = nn.functional.relu(self.input_proj(mel_frames))
        out = nn.functional.dropout(out, 0.5, True) # force dropout even during inference 
        out =  nn.functional.relu(self.out_proj(out))
        out = nn.functional.dropout(out, 0.5, True) # force dropout even during inference 
        return out

class PostNet(nn.Module):
    def __init__(self, mel_bins, post_net_channels, post_net_kernel_size, num_layers=3):
        super(PostNet, self).__init__()
        self.convolutions = nn.ModuleList()
        assert num_layers > 2, 'Number of convolutions in post net should be > 2'
        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(mel_bins, post_net_channels,
                        kernel_size=post_net_kernel_size, stride=1,
                        padding=int((post_net_kernel_size - 1) / 2)),
                nn.BatchNorm1d(post_net_channels))
        )

        for _ in range(1, num_layers - 1):
            self.convolutions.append(
                nn.Sequential(
                    nn.Conv1d(post_net_channels, post_net_channels,
                            kernel_size=post_net_kernel_size, stride=1,
                            padding=int((post_net_kernel_size - 1) / 2)),
                    nn.BatchNorm1d(post_net_channels))
            )

        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(post_net_channels, mel_bins,
                        kernel_size=post_net_kernel_size, stride=1,
                        padding=int((post_net_kernel_size - 1) / 2)),
                nn.BatchNorm1d(mel_bins))
        )
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = self.dropout(torch.tanh(self.convolutions[i](x)))
        x = self.dropout(self.convolutions[-1](x)) # no tanh for last layer
        return x

# from pytorch documentation https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, embedding_size: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2) * (-math.log(10000.0) / embedding_size))
        pe = torch.zeros(max_len, 1, embedding_size)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
            
        """
        x = x.permute(1,0,2)
        x = x + scale*self.pe[:x.size(0)]
        x = self.dropout(x).permute(1,0,2)
        return x


class TransformerTTS_No_PE(nn.Module):
    def __init__(self, vocab_size, encoder_embedding_size, embedding_size, ff_hidden_size, 
                 post_net_channels, post_net_kernel_size, mel_bins=80, dropout_encoder=0.2):
        super(TransformerTTS_No_PE, self).__init__()
        self.mel_bins = mel_bins
        # -----Positional Encoding-----
        self.positional_encoding = nn.Embedding(embedding_size)
        self.encoder_scale = nn.Parameter(torch.ones(1))
        self.decoder_scale = nn.Parameter(torch.ones(1))
        # -----Encoder-----
        self.encoder_pre_net = EncoderPreNet(vocab_size, encoder_embedding_size, embedding_size, dropout=dropout_encoder)
        self.encoder_block_1 = EncoderBlock(embedding_size, ff_hidden_size)
        self.encoder_block_2 = EncoderBlock(embedding_size, ff_hidden_size)
        self.encoder_block_3 = EncoderBlock(embedding_size, ff_hidden_size)
        # -----Decoder-----
        self.decoder_pre_net = DecoderPreNet(mel_bins, embedding_size)
        self.decoder_block_1 = DecoderBlock(embedding_size, ff_hidden_size)
        self.decoder_block_2 = DecoderBlock(embedding_size, ff_hidden_size)
        self.decoder_block_3 = DecoderBlock(embedding_size, ff_hidden_size)
        # -----Projection-----
        self.lin_proj = nn.Linear(embedding_size, mel_bins) # mel projection
        self.eos_gate = nn.Linear(embedding_size, 1) # stop projection
        # -----Post Net-----
        self.post_net = PostNet(mel_bins, post_net_channels, post_net_kernel_size, num_layers=5)

    
    @staticmethod
    def get_padding_mask(batch_size, text_lens, max_text_len, mel_lens, max_mel_len, device):
        text_pad_mask = torch.zeros(batch_size, max_text_len, device=device).masked_fill(
            get_mask_from_lens(text_lens, max_text_len).to(device), float("-inf"))
        mel_pad_mask = torch.zeros(batch_size, max_mel_len, device=device).masked_fill(
            get_mask_from_lens(mel_lens, max_mel_len).to(device), float("-inf"))
        return text_pad_mask, mel_pad_mask
    
    @staticmethod
    def get_attn_mask(max_text_len, max_mel_lens, device):
        text_self_mask = torch.zeros(max_text_len, max_text_len, device=device).masked_fill(
            torch.triu(torch.full((max_text_len, max_text_len), True, device=device), diagonal=1),
            float('-inf'))
        mel_self_mask = torch.zeros(max_mel_lens, max_mel_lens, device=device).masked_fill(
            torch.triu(torch.full((max_mel_lens, max_mel_lens), True, device=device), diagonal=1),
            float('-inf'))
        look_ahead_mask = torch.zeros(max_mel_lens, max_text_len, device=device).masked_fill(
            torch.triu(torch.full((max_mel_lens, max_text_len), True, device=device), diagonal=1),
            float('-inf'))
        return text_self_mask, mel_self_mask, look_ahead_mask

    def mask_output(self, mel_outputs: torch.Tensor, post_net_output: torch.Tensor, gate_outputs: torch.Tensor, 
                    mel_spec_lens: torch.Tensor, max_mel_len):
        mask = get_mask_from_lens(mel_spec_lens, max_mel_len).to(mel_outputs.device)
        masked_mel_outputs = mel_outputs.masked_fill(mask.unsqueeze(-1), 0)
        masked_post_output = post_net_output.masked_fill(mask.unsqueeze(-1), 0)
        masked_gate_outputs = gate_outputs.masked_fill(mask.unsqueeze(-1), 1e3) # Sigmoid will convert masked logit to probability ≈ 1.0   
        masked_gate_outputs = masked_gate_outputs.squeeze(-1)
        return masked_mel_outputs, masked_post_output, masked_gate_outputs, mask

    def get_decoder_sos(self, batch_size):
        return torch.zeros(batch_size, self.mel_bins) # batch, n_mels


    def forward(self, text: torch.Tensor, text_seq_lens: torch.Tensor, mel_spec_true: torch.Tensor, 
                mel_spec_lens: torch.Tensor):
        max_text_len = text.size(1)
        batch_size = text.size(0)
        # add SOS token to mel
        sos = self.get_decoder_sos(batch_size).unsqueeze(1).to(mel_spec_true.device)
        mel_spec_true = torch.cat((sos, mel_spec_true), dim=1)
        max_mel_len = mel_spec_true.size(1)
        
        text_pad_mask, mel_pad_mask = self.get_padding_mask(batch_size, text_seq_lens, max_text_len, 
                                                            mel_spec_lens, max_mel_len, text.device)
        text_self_mask, mel_self_mask, look_ahead_mask = self.get_attn_mask(max_text_len, max_mel_len, text.device)

        # get embeddings
        text = self.encoder_pre_net(text)
        # add positional encoding to text
        text = self.positional_encoding(text, self.encoder_scale)

        # ENCODER
        text = self.encoder_block_1(text, attn_mask=text_self_mask, key_padding_mask=text_pad_mask)
        text = self.encoder_block_2(text, attn_mask=text_self_mask, key_padding_mask=text_pad_mask)
        text = self.encoder_block_3(text, attn_mask=text_self_mask, key_padding_mask=text_pad_mask)

        # DECODER PRE_NET
        mel = self.decoder_pre_net(mel_spec_true)
        # add positional encoding to mel specs
        mel = self.positional_encoding(mel, self.decoder_scale)

        # DECODER
        mel, _ = self.decoder_block_1(mel, text, mel_self_mask=mel_self_mask, mel_pad_mask=mel_pad_mask, 
                                   look_ahead_mask=look_ahead_mask, text_pad_mask=text_pad_mask)
        mel, _ = self.decoder_block_2(mel, text, mel_self_mask=mel_self_mask, mel_pad_mask=mel_pad_mask, 
                                   look_ahead_mask=look_ahead_mask, text_pad_mask=text_pad_mask)
        mel, attn_weights = self.decoder_block_3(mel, text, mel_self_mask=mel_self_mask, mel_pad_mask=mel_pad_mask, 
                                   look_ahead_mask=look_ahead_mask, text_pad_mask=text_pad_mask)

        # remove sos token from starting frame (consistent with seq2seqTTS)
        mel = mel[:,1:,:]
        max_mel_len = max_mel_len - 1

        # PROJECTION
        mel_proj = self.lin_proj(mel)
        stop_token = self.eos_gate(mel)
        
        # POST NET
        mel_post_net = self.post_net(mel_proj.permute(0, 2, 1))
        mel_post_net = mel_post_net.permute(0, 2, 1)

        mel_post_net = mel_post_net + mel_proj # added residual connection

        masked_mel_outputs, masked_post_output, masked_gate_outputs, mask = self.mask_output(mel_proj, mel_post_net, stop_token, 
                                                                                            mel_spec_lens, max_mel_len)
        
        # need to get attention outputs as well
        return masked_mel_outputs, masked_post_output, masked_gate_outputs, attn_weights, mask


    # this is identical to forward except it does not add an sos token to the start
    # here mel_spec_true will always have sos, [sos] -> [sos, pred1] -> [sos, pred1, pred2]
    # there are better ways to do this, but will take some refactoring for compatibility for tacotron2mini and seq2seq
    def inference_forward(self, text: torch.Tensor, text_seq_lens: torch.Tensor, mel_spec_true: torch.Tensor, 
                mel_spec_lens: torch.Tensor):
        max_text_len = text.size(1)
        max_mel_len = mel_spec_true.size(1)
        batch_size = text.size(0)
        text_pad_mask, mel_pad_mask = self.get_padding_mask(batch_size, text_seq_lens, max_text_len, 
                                                            mel_spec_lens, max_mel_len, text.device)
        text_self_mask, mel_self_mask, look_ahead_mask = self.get_attn_mask(max_text_len, max_mel_len, text.device)

        # get embeddings
        text = self.encoder_pre_net(text)
        # add positional encoding to text
        text = self.positional_encoding(text)

        # ENCODER
        text = self.encoder_block_1(text, attn_mask=text_self_mask, key_padding_mask=text_pad_mask)
        text = self.encoder_block_2(text, attn_mask=text_self_mask, key_padding_mask=text_pad_mask)
        text = self.encoder_block_3(text, attn_mask=text_self_mask, key_padding_mask=text_pad_mask)

        # DECODER PRE_NET
        mel = self.decoder_pre_net(mel_spec_true)
        # add positional encoding to mel specs
        mel = self.positional_encoding(mel)

        # DECODER
        mel, _ = self.decoder_block_1(mel, text, mel_self_mask=mel_self_mask, mel_pad_mask=mel_pad_mask, 
                                   look_ahead_mask=look_ahead_mask, text_pad_mask=text_pad_mask)
        mel, _ = self.decoder_block_2(mel, text, mel_self_mask=mel_self_mask, mel_pad_mask=mel_pad_mask, 
                            look_ahead_mask=look_ahead_mask, text_pad_mask=text_pad_mask)
        mel, attn_weights = self.decoder_block_3(mel, text, mel_self_mask=mel_self_mask, mel_pad_mask=mel_pad_mask, 
                                   look_ahead_mask=look_ahead_mask, text_pad_mask=text_pad_mask)

        # PROJECTION
        mel_proj = self.lin_proj(mel)
        stop_token = self.eos_gate(mel)
        
        # POST NET
        mel_post_net = self.post_net(mel_proj.permute(0, 2, 1))
        mel_post_net = mel_post_net.permute(0, 2, 1)

        mel_post_net = mel_post_net + mel_proj # added residual connection

        masked_mel_outputs, masked_post_output, masked_gate_outputs, mask = self.mask_output(mel_proj, mel_post_net, stop_token, 
                                                                                            mel_spec_lens, max_mel_len)
        
        # need to get attention outputs as well
        return masked_mel_outputs, masked_post_output, masked_gate_outputs, attn_weights, mask


    @torch.no_grad()
    def inference(self, text, device, max_length=800, mel_bins=80, stop_token_threshold = 0.5, with_tqdm=False):
        self.eval()    
        self.train(False)
        text_lengths = torch.tensor(text.shape[1]).unsqueeze(0).to(device)
        N = 1
        SOS = torch.zeros((N, 1, mel_bins)).to(device)
        
        mel = SOS
        mel_lengths = torch.tensor(1).unsqueeze(0).to(device)
        stop_token_outputs = torch.FloatTensor([]).to(device)

        _range = range(max_length)
        if with_tqdm:
            _range = tqdm(range(max_length))

        for i in _range:
            _, masked_post_output, gate_outputs, _, _ = self.inference_forward(text, text_lengths, mel, mel_lengths)
            mel = torch.cat([mel, masked_post_output[:, -1:, :]], dim=1)

            if torch.sigmoid(gate_outputs[:,-1]) > stop_token_threshold:      
                break
            else:
                if i == max_length-1:
                    print("Max inference length reached.")
                    break
                stop_token_outputs = torch.cat([stop_token_outputs, gate_outputs[:,-1:]], dim=1)
                mel_lengths = torch.tensor(mel.size(1)).unsqueeze(0)

        mel = mel[:,1:,:] # remove sos
        return mel, stop_token_outputs