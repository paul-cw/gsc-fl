# Directory to download the google speech command dataset to
data_dir = './data/'

# Desired no of clients/silos/partitions. E.g. [2,4] will create partitions 
# into 2 and 4 clients and add them as a column to the dataframe
n_silos = [2, 4, 8, 16, 32, 64, 128, 256, 512]

# Path to save the dataframe that is ready to use for FL
save_to_path = './df_gsc_fl_generated.csv'
