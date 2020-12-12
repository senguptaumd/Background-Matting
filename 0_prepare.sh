#!/bin/bash

# Load tensorflow models (we need DeepLabV3+ for coarse people segmentation)
git clone https://github.com/tensorflow/models.git

# Create a docker image with all dependencies
docker build -t backmatting -f dockerfile .

# Download pretrained models
wget https://gist.githubusercontent.com/andreyryabtsev/458f7450c630952d1e75e195f94845a0/raw/0b4336ac2a2140ac2313f9966316467e8cd3002a/download.sh
chmod +x download.sh
./download.sh
