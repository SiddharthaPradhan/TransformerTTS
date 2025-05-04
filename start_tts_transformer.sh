#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=50g
#SBATCH -J "TTS Transformer"
#SBATCH -p short
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/home/sppradhan/TransformerTTS/transformer_tts_%j.txt

module load cuda
module load python/3.12.6
source ~/TransformerTTS/.venv/bin/activate
python main.py