# README
This repo contains the GSC-FL dataset, derived from the Google Speech Commands [dataset](https://arxiv.org/abs/1804.03209v1). It is a **keyword spotting dataset specifically designed for federated learning**. It containts partitions on 2,4,8,...,512 clients of the original GSC-FL dataset and more can be created. The partition strategy ensures that a **given speaker is unique to one client**. Baselines partitions (iid) are provided as well. 

## Requirements
- Linux (potentially Mac)
- wget is installed for downloading the dataset

## Installation
- create e.g. a conda environment, via `conda create --name FL python=3.8`
- run `conda activate FL`
- install the dependencies listed in requirements.txt, via:  `pip install -r requirements.txt`


## Run the code

You can **create your own splits**:
- set the n_splits parameter in config.py and run `python make_custom_split.py`. (This downloads the audio files if they are not downloaded already). This will create a dataframe, which is called **df_gsc_fl_generated.csv**. Use `pd.read_csv('./df_gsc_fl_generated.csv')` inside your python project.

To **reprdoduce the results from the paper**:
- download the audio files via `python download_gsc.py` and use `pd.read_csv('./df_gsc_fl_default.csv')` inside your python project.


## What's inside the dataframe?
Each row in the dataframe corresponds to an instance and the columns contain:
- **file_path**: path to the audio file 
- **keyword**: keyword that is uttered in the audio file
- **dataset**: training, testing, validation split
- **speaker_id**: unique id for each speaker
- **speaker_ut**: utterance count of this speaker (starts at 0)
- **label_one_hot**: one hot encoded keyword label
- **n2_kk**: id for split into 2 clients using the kk method (given speaker is unique to one client)
- **n8_random**: id for split into 8 clients using random splitting (given speaker is not unique to one client )
- and so on...



