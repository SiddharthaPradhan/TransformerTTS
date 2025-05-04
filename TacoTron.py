import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchaudio.transforms as T

def get_mask_from_lens(lens, max_len):
    base_mask = torch.arange(max_len).expand(lens.size(0), max_len).T
    mask = (base_mask > lens).permute(1,0)
    return mask


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, dropout=0.0):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.enc_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.enc_conv1 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, padding=1, stride=1, dilation=1)
        self.enc_batch_norm_1 = nn.BatchNorm1d(
            embedding_dim
        )
        self.enc_dropout_1 = nn.Dropout(dropout)
        self.enc_conv2 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, padding=1, stride=1, dilation=1)
        self.enc_batch_norm_2 = nn.BatchNorm1d(
            embedding_dim
        )
        self.enc_dropout_2 = nn.Dropout(dropout)
        self.enc_rnn = nn.GRU(embedding_dim, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor, text_seq_lens: torch.Tensor):
        # ENCODER
        x = self.enc_embedding(x) # batch_size, max_seq_len, embedding_dim
        x = x.permute(0, 2, 1)
        # First Conv Layer
        x = self.enc_conv1(x)
        x = self.enc_batch_norm_1(x)
        x = nn.functional.relu(x)
        x = self.enc_dropout_1(x)
        # Second Conv Layer
        x = self.enc_conv2(x)
        x = self.enc_batch_norm_2(x)
        x = nn.functional.relu(x)
        x = self.enc_dropout_2(x)
        x = x.permute(0, 2, 1)
        packed_x = pack_padded_sequence(x , text_seq_lens.cpu().numpy(), batch_first=True) 
        output, hidden = self.enc_rnn(packed_x)
        output, _ = pad_packed_sequence(output, batch_first=True)
        return output, hidden

class LocationLayer(nn.Module):
    def __init__(self, attention_dim,  n_filters=32, location_kernel_size=32):
        super(LocationLayer, self).__init__()
        valid_padding = int((location_kernel_size - 1) / 2)
        self.conv = nn.Conv1d(2, n_filters, location_kernel_size, stride=1, padding=valid_padding, bias=False)
        self.location_proj = nn.Linear(n_filters, attention_dim)
        torch.nn.init.xavier_uniform_(
            self.location_proj.weight,
            gain=torch.nn.init.calculate_gain('tanh'))
        torch.nn.init.xavier_uniform_(
            self.conv.weight,
            gain=torch.nn.init.calculate_gain('tanh'))

    def forward(self, attention_weights_cat):
        processed_attention = self.conv(attention_weights_cat) 
        processed_attention = processed_attention.transpose(1, 2) 
        processed_attention = self.location_proj(processed_attention) 
        return processed_attention

# Bahdanau Attention (Additive) + Location Layer
class LocationSensitiveAttention(nn.Module):
    def __init__(self, dec_hidden_size, enc_hidden_size, attention_dim,  n_filters=32, location_kernel_size=32):
        super(LocationSensitiveAttention, self).__init__()
        self.dec_hidden_size = dec_hidden_size # hidden size of the decoder (first decoder layer if l > 1)
        self.enc_hidden_size = enc_hidden_size # hidden size of the encoder (hidden states from both directions)
        self.query_proj = nn.Linear(dec_hidden_size, attention_dim, bias=False) # project decoder hidden to attention dim
        self.keys_proj = nn.Linear(enc_hidden_size*2, attention_dim, bias=False)  # project all hidden encoder states to attention dim
        self.score_proj = nn.Linear(attention_dim, 1) # alignment score
        self.location = LocationLayer(attention_dim, n_filters=n_filters, location_kernel_size=location_kernel_size)
        self.init_weights() # optimize gain for the activation function used

    def init_weights(self):
        # passed to tanh
        torch.nn.init.xavier_uniform_(
            self.query_proj.weight,
            gain=torch.nn.init.calculate_gain('tanh'))
        # passed to tanh
        torch.nn.init.xavier_uniform_(
            self.keys_proj.weight,
            gain=torch.nn.init.calculate_gain('tanh'))
        # final linear proj
        torch.nn.init.xavier_uniform_(
            self.score_proj.weight,
            gain=torch.nn.init.calculate_gain('linear'))

    def get_score(self, query: torch.Tensor, keys_projected: torch.Tensor, att_weights_combined: torch.Tensor) -> torch.Tensor:
        projected_query = self.query_proj(query)
        processed_location = self.location(att_weights_combined)
        scores = self.score_proj(torch.tanh(projected_query + keys_projected + processed_location))
        return scores


    def forward(self, query: torch.Tensor, keys_projected: torch.Tensor, att_weights_combined: torch.Tensor, key_mask: torch.Tensor):
        # get scores
        scores = self.get_score(query, keys_projected, att_weights_combined)
        # mask scores for padding
        if key_mask is not None:
            scores = scores.masked_fill(key_mask.unsqueeze(2), -1*float('inf'))
        # calculate weights and context
        weights = nn.functional.softmax(scores, dim=1).permute(0,2,1)
        context = torch.bmm(weights, keys_projected)
        # return context (used by decoder), and weights (for cumulative weights) for next location layer
        return context, weights.squeeze(1)
    
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

class Decoder(nn.Module):
    def __init__(self, pre_net_out_size, decoder_hidden_size=512, enc_hidden_size = 512, 
                n_filters=32, location_kernel_size=32, attn_dim=128, mel_bins=128,  dropout=0.1):
        super(Decoder, self).__init__()
        self.mel_bins = mel_bins
        self.decoder_hidden_size = decoder_hidden_size
        self.enc_hidden_size = enc_hidden_size
        self.dropout = nn.Dropout(dropout)
        self.pre_net = DecoderPreNet(mel_bins, pre_net_out_size)
        # layer 1 rnn that uses attention
        self.attn_rnn = nn.GRU(pre_net_out_size+enc_hidden_size, decoder_hidden_size, batch_first=True)
        # attention module
        self.attention = LocationSensitiveAttention(decoder_hidden_size, enc_hidden_size, 
                                                    attn_dim, n_filters, location_kernel_size)
        # layer 2 rnn for pre-projected output using attended input
        self.out_rnn = nn.GRU(enc_hidden_size+decoder_hidden_size, decoder_hidden_size, batch_first=True)
        self.dec_lin_proj = nn.Linear(decoder_hidden_size+enc_hidden_size, mel_bins) # mel projection
        self.dec_eos_gate = nn.Linear(decoder_hidden_size+enc_hidden_size, 1) # stop projection
        # attention weights are initialized in init_decoder
        self.attention_weights = None # attention current time step
        self.attention_cumulative = None # cumulative attention weights (sum of previous weights)
        self.keys_projected = None # projected encoder outputs

    def init_decoder_states(self, encoder_out: torch.Tensor):
        # initialize hidden state with learned parameters
        batch_size = encoder_out.size(0)
        max_encoder_time = encoder_out.size(1)
        
        self.attn_rnn_hidden =  encoder_out.new_zeros(batch_size, self.decoder_hidden_size).unsqueeze(0)
        self.out_rnn_hidden = encoder_out.new_zeros(batch_size, self.decoder_hidden_size).unsqueeze(0)
        # initialize attention weights
        self.attention_weights = encoder_out.new_zeros(batch_size, max_encoder_time)
        self.attention_cumulative = encoder_out.new_zeros(batch_size, max_encoder_time)
        self.attn_context =  encoder_out.new_zeros(batch_size, self.enc_hidden_size).unsqueeze(1)
        self.keys_projected = self.attention.keys_proj(encoder_out) # need to calculate once per batch
        
        
    def step(self, decoder_input, key_mask):
        """Process a single time step
        Args:
            decoder_input (nn.Tensor): output of prev time step (could be teacher forced)
        """
        # First layer rnn with attention
        attn_rnn_input = torch.cat((decoder_input, self.attn_context), dim=-1) # along feature dim
        _, self.attn_rnn_hidden = self.attn_rnn(attn_rnn_input, self.attn_rnn_hidden)
        self.attn_rnn_hidden = self.dropout(self.attn_rnn_hidden)
        location_attn_input = torch.cat((self.attention_weights.unsqueeze(1), self.attention_cumulative.unsqueeze(1)), dim=1)
        self.attn_context, self.attention_weights = self.attention(self.attn_rnn_hidden.permute(1,0,2), 
                                                                   self.keys_projected, location_attn_input, key_mask)
        self.attention_cumulative += self.attention_weights
        # second later rnn with attended input
        out_rnn_input = torch.cat((self.attn_rnn_hidden.permute(1,0,2), self.attn_context), dim=-1)
        _, self.out_rnn_hidden = self.out_rnn(out_rnn_input, self.out_rnn_hidden)
        self.out_rnn_hidden = self.dropout(self.out_rnn_hidden)
        attended_out_hidden = torch.cat((self.out_rnn_hidden.permute(1,0,2), self.attn_context), dim=-1)
        # output mel frame and stop token
        mel_out = self.dec_lin_proj(attended_out_hidden)
        stop_out = self.dec_eos_gate(attended_out_hidden)
        return mel_out, stop_out, self.attention_weights
        
          
    def forward(self, enc_outputs: torch.Tensor, text_lens: torch.Tensor, max_text_len: int, 
                mel_spec_true: torch.Tensor, mel_spec_lens: torch.Tensor, teacher_force_ratio=0.0):
        self.init_decoder_states(enc_outputs) # initialize hidden layers and keys for attention
        enc_mask = get_mask_from_lens(text_lens, max_text_len).to(enc_outputs.device)
        dec_input = self.pre_net(self.get_decoder_sos(mel_spec_true)).unsqueeze(1) # batch, 1, n_mels (SOS)
        pre_net_out = self.pre_net(mel_spec_true)
        mel_outputs = []
        gate_outputs = []
        attention_outputs = []
        # iterate time_dim (i.e. max_mel_length of batch)
        max_time = mel_spec_true.size(1)
        for t_step in range(max_time):
            mel_out, stop_out, attention_weights = self.step(dec_input, enc_mask)
            mel_outputs.append(mel_out.squeeze(1))
            gate_outputs.append(stop_out.squeeze(1))
            attention_outputs.append(attention_weights.squeeze(1))
            # teacher forcing (y has shape: batch_size, mel_seq_len, n_mels)
            if torch.rand(1).item() < teacher_force_ratio: # use teacher forcing?
                dec_input = pre_net_out[:, t_step:t_step+1, :] # use output from actual sequence
            else:
                dec_input = self.pre_net(mel_out)  # use output from pred of decoder
        mel_outputs = torch.stack(mel_outputs, dim=1)
        gate_outputs =  torch.stack(gate_outputs, dim=1).squeeze(-1)
        attention_outputs =  torch.stack(attention_outputs, dim=1).squeeze(-1)
        masked_mel_outputs, masked_gate_outputs, mask = self.mask_output(mel_outputs, gate_outputs, mel_spec_lens, max_time)
        return masked_mel_outputs, masked_gate_outputs, attention_outputs, mask

    def get_decoder_sos(self, mel_spec_true: torch.Tensor):
        return mel_spec_true.new_zeros(mel_spec_true.size(0), mel_spec_true.size(2)) # batch, 1, n_mels

    def mask_output(self, mel_outputs: torch.Tensor, gate_outputs: torch.Tensor, mel_spec_lens: torch.Tensor, max_mel_len):
        mask = get_mask_from_lens(mel_spec_lens, max_mel_len).to(mel_outputs.device)
        masked_mel_outputs = mel_outputs.masked_fill(mask.unsqueeze(-1), 0)
        masked_gate_outputs = gate_outputs.masked_fill(mask, 1e3) # Sigmoid will convert masked logit to probability ≈ 1.0   
        return masked_mel_outputs, masked_gate_outputs, mask

class TacoTronTTS(nn.Module):
    def __init__(self, vocab_size, embedding_dim, enc_hidden_size, decoder_hidden_size, 
                 pre_net_out_size, location_n_filters, location_kernel_size, attn_dim, 
                 post_net_channels, post_net_kernel_size,
                 dropout_encoder=0.2, dropout_decoder=0.1, mel_bins=80):
        super(TacoTronTTS, self).__init__()
        self.mel_bins = mel_bins
        # -----Encoder-----
        self.encoder = Encoder(vocab_size, embedding_dim, enc_hidden_size, 
                               dropout=dropout_encoder)
        # -----Decoder-----
        self.decoder = Decoder(pre_net_out_size, decoder_hidden_size, enc_hidden_size, 
                               location_n_filters, location_kernel_size, attn_dim, 
                               mel_bins, dropout_decoder)
        
        # -----PostNet-----
        self.post_net = PostNet(mel_bins, post_net_channels, post_net_kernel_size)

    def forward(self, x: torch.Tensor, text_seq_lens: torch.Tensor, mel_spec_true: torch.Tensor, 
                mel_spec_lens: torch.Tensor, teacher_force_ratio=0.0):
        encoder_outputs, _  = self.encoder(x, text_seq_lens)
        max_text_len = x.size(1)
        # DECODER
        masked_mel_outputs, masked_gate_outputs, attention_outputs, mask = self.decoder(encoder_outputs, text_seq_lens, max_text_len, mel_spec_true, mel_spec_lens, teacher_force_ratio)
        post_net_outputs = self.post_net(masked_mel_outputs.permute(0, 2, 1))
        post_net_outputs = post_net_outputs.permute(0, 2, 1)
        post_net_outputs = masked_mel_outputs + post_net_outputs # added residual (skipped connection)
        post_net_outputs = post_net_outputs.masked_fill(mask.unsqueeze(-1), 0)
        return masked_mel_outputs, post_net_outputs, masked_gate_outputs, attention_outputs, mask
   

    @torch.no_grad()
    # TODO: FIXME Update with new structure
    def inference(self, text_seq, max_mel_length=800, stop_token_thresh=0.5):
        self.train(False)
        self.eval()
        # assume text_seq has shape (1, input_seq_len)
        text_lengths = torch.IntTensor([text_seq.shape[-1]])
        # TODO: separate encoder and decoder logic into different classes
        #       to reduce repetition like below.
        x = self.enc_embedding(text_seq) # 1, max_seq_len, embedding_dim
        x = x.permute(0, 2, 1)
        # First Conv Layer
        x = self.enc_conv1(x)
        x = self.enc_batch_norm_1(x)
        x = nn.functional.relu(x)
        x = self.enc_dropout_1(x)
        # Second Conv Layer
        x = self.enc_conv2(x)
        x = self.enc_batch_norm_2(x)
        x = nn.functional.relu(x)
        x = self.enc_dropout_2(x)
        x = x.permute(0, 2, 1)
        packed_x = pack_padded_sequence(x , text_lengths.cpu().numpy(), batch_first=True, enforce_sorted=False) 
        _, (hidden, cell) = self.enc_lstm(packed_x)
        hidden = torch.cat(hidden.unbind(), dim=1).unsqueeze(0)
        cell = torch.cat(cell.unbind(), dim=1).unsqueeze(0)
        inference_batch_size = 1
        dec_input = self.get_decoder_sos(inference_batch_size, self.mel_bins) # (1, mel_bins)
        mel_outputs = []
        for t_step in range(max_mel_length):
            dec_lstm_output, (hidden, cell) = self.dec_lstm(dec_input, (hidden, cell))
            # pass output of decoder lstm to fc layer to map from dec_lstm_hidden to n_mels
            dec_mel_output = self.dec_lin_proj(dec_lstm_output)
            mel_outputs.append(dec_mel_output.squeeze(1))
            # linear projection of lstm out to mel_dims
            dec_gate_output = self.dec_eos_gate(dec_lstm_output)
            # teacher forcing (y has shape: batch_size, mel_seq_len, n_mels)
            # if network predicted stop token
            if torch.nn.functional.sigmoid(dec_gate_output) > stop_token_thresh:
                break
            if t_step == max_mel_length-1:
                print('WARNING: max_mel_length (800) reached, model wanted to generate longer speech.')

        mel_outputs = torch.stack(mel_outputs, dim=1).to(self.device)
        return mel_outputs