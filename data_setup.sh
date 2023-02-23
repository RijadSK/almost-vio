#!/bin/bash

# Install dependency
sudo apt-get install p7zip-full


mkdir data
cd data

GREEN='\033[0;32m'
NC='\033[0m' # No Color


# Downlad all 23 scenes of ADVIO dataset
echo -e "\n\n${GREEN}DOWNLOADING THE DATASET${NC}"
for i in $(seq -f "%02g" 1 23);
do
    # Download
    wget -O advio-$i.zip https://zenodo.org/record/1476931/files/advio-$i.zip

    # Extract
    7z x advio-$i.zip

    # Cleaning
    rm advio-$i.zip
done

# Extracting frames from video scenes
echo -e "\n\n${GREEN}EXTRACTING FRAMES FROM VIDEOS${NC}"
python ../video_to_frames.py

# Preparing the data for the training
echo -e "\n\n${GREEN}PREPARING THE DATA${NC}"
python ../data_preprocessing.py