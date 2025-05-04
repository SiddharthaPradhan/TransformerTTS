import torch
import torchaudio
from torchaudio.functional import spectrogram
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pylab as plt

# From NVIDIA TacoTron2 params
sr = 22050
n_fft = 2048
n_stft = int((n_fft//2) + 1)

frame_shift = 0.0125 # seconds
hop_length = int(n_fft/8.0)

frame_length = 0.05 # seconds  
win_length = int(n_fft/2.0)

max_mel_time = 1024

max_db = 100  
scale_db = 10
ref = 4.0
power = 2.0
norm_db = 10 
ampl_multiplier = 10.0
ampl_amin = 1e-10
db_multiplier = 1.0
ampl_ref = 1.0
ampl_power = 1.0

class SpeechConverter():
    def __init__(self, num_mels):
        self.num_mel = num_mels
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, 
            win_length=win_length,
            hop_length=hop_length,
            power=power
        )
        self.mel_scale_transform = torchaudio.transforms.MelScale(
            n_mels=self.num_mel, 
            sample_rate=sr, 
            n_stft=n_stft
        )

        self.mel_inverse_transform = torchaudio.transforms.InverseMelScale(
            n_mels=self.num_mel, 
            sample_rate=sr, 
            n_stft=n_stft
        )

        self.griffnlim_transform = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
        )
        
    def pow_to_db_mel_spec(self,mel_spec):
        mel_spec = torchaudio.functional.amplitude_to_DB(
            mel_spec,
            multiplier = ampl_multiplier, 
            amin = ampl_amin, 
            db_multiplier = db_multiplier, 
            top_db = max_db
        )
        mel_spec = mel_spec/scale_db
        return mel_spec

    def convert_to_mel_spec(self, raw_audio):
        spec = self.spec_transform(raw_audio)
        mel_spec = self.mel_scale_transform(spec)
        db_mel_spec = self.pow_to_db_mel_spec(mel_spec)
        db_mel_spec = db_mel_spec.squeeze(0)
        return db_mel_spec
    
    def inverse_mel_spec_to_wav(self, mel_spec):
        power_mel_spec = self.db_to_power_mel_spec(mel_spec)
        spectrogram = self.mel_inverse_transform(power_mel_spec)
        pseudo_wav = self.griffnlim_transform(spectrogram)
        return pseudo_wav

    def db_to_power_mel_spec(self, mel_spec):
        mel_spec = mel_spec*scale_db
        mel_spec = torchaudio.functional.DB_to_amplitude(
            mel_spec,
            ref=ampl_ref,
            power=ampl_power
            )  
        return mel_spec
    
    


# plotting utils from https://github.com/BogiHsu/Tacotron2-PyTorch/blob/master/utils/plot.py
def figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data.transpose(2, 0, 1)


def alignment_to_numpy(alignment, info=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment, aspect='auto', origin='lower',
                   interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()

    fig.canvas.draw()
    data = figure_to_numpy(fig)
    plt.close()
    return data


def spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = figure_to_numpy(fig)
    plt.close()
    return data