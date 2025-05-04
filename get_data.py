from datasets import load_dataset
import os


def save_lj_speech_dataset(save_dir='./data/'):
    os.makedirs(save_dir, exist_ok = True) # if dir not made, make it else nothing
    hf_dataset = load_dataset('keithito/lj_speech')
    hf_dataset.save_to_disk(save_dir)
    print('Saved LJ Speech Dataset to disk')

if __name__ == "__main__":
    save_lj_speech_dataset()