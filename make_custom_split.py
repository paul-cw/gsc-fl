"""
Script to perform custom splitting of the GSC_FL_dataset. The resulting dataset is saved to disk.
The path to the actual .wav files is contained.
"""

from config import data_dir, n_silos, save_to_path
import random
from utilities import download_gsc, gsc_to_dataframe_without_audios, partition_and_save_dataset
import numpy as np

if __name__ == "__main__":
    # Fix the random seed
    random.seed(0)
    np.random.seed(0)

    # Download GSC dataset 
    print("----- Start downloading GSC dataset:\n")
    download_gsc(data_dir=data_dir)

    # Prepare the central dataset into a pd dataframe
    print("----- Start preparing dataframes:\n")
    df_gsc_fl = gsc_to_dataframe_without_audios(data_dir=data_dir)

    # Partition the dataset for FL
    print("----- Start partitioning the dataset:\n")
    df_gsc_fl = partition_and_save_dataset(df_gsc_fl, n_silos=n_silos, part_alg='kk', save_to_path = save_to_path)
    df_gsc_fl = partition_and_save_dataset(df_gsc_fl, n_silos=n_silos, part_alg='random', save_to_path = save_to_path)
