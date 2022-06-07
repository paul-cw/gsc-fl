"""Script to download GSC dataset and use existing default FL CSV file 
available without custom split.
"""
from config import data_dir
from utilities import download_gsc

if __name__ == "__main__":
    download_gsc(data_dir=data_dir)

