#!/bin/bash
#SBATCH --job-name=movienet-dl
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/download_%j.log

set -e
mkdir -p $HOME/movienet
cd $HOME/movienet

if [ ! -d bassl ]; then
    git clone https://github.com/kakaobrain/bassl.git
fi

cd bassl
bash scripts/download_movienet_data.sh
echo "download finished at $(date)"
