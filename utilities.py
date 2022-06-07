"""Script that performs tasks and sub-tasks for partitioning.
"""

import numpy as np
import os
from os import fspath
from pathlib import Path
import soundfile as soundfile_
import pandas as pd
import tqdm
from numberpartitioning import karmarkar_karp


def download_gsc(data_dir: str):
    """Helper function to download, unpack and prepare the GSC dataset.
    Downloads the dataset if one doesn't already exist in the given 
    directory, else ignores it. Also, creates a silence audio file in the same
    directory if one doesn't exist yet.

    Parameters
    ----------
    data_dir: str
        Directory to download and prepare dataset.

    Returns
    -------
        None
    """
    print('try loading dataset from ', data_dir)

    if not os.path.isdir(data_dir):
        print('Loading dataset')
        os.mkdir(data_dir)
        print('Start downloading')
        os.system(
            f"wget -O {data_dir}speech_commands_v0.02.tar.gz http://download.tensorflow.org/data/speech_commands_v0"
            f".02.tar.gz")
        print('Start unzipping')
        os.system('tar -xf %sspeech_commands_v0.02.tar.gz -C ' % data_dir + '%s' % data_dir[0:-1])
        print('Start removing .zip file')
        os.system('rm %sspeech_commands_v0.02.tar.gz' % data_dir)
        print('Done!')
    else:
        print('Dataset is already downloaded!')

    # Create a 'silence' file for the 'silence' class
    fs = 16000
    if not os.path.exists(data_dir + 'silence/silence.wav'):
        if not os.path.exists(data_dir + 'silence/'):
            os.mkdir(data_dir + 'silence/')
        soundfile_.write(data_dir + 'silence/silence.wav', np.zeros(shape=fs), fs)


def one_at(x: int, n_output_neurons: int = 12):
    """Helper function for encoding label integers as one-hot encoded

    Parameters
    ----------
    x: Any
        Labels to encode.
    n_output_neurons: int, optional
        Number of output neurons describing the output keywords. Defaults to `12`.

    Returns
    -------
    target_labels: list[float]
        One-hot encoded labels        
    """
    target_labels = [0.] * n_output_neurons
    target_labels[x] = 1.
    return target_labels


def create_training_list(validation_file: str, testing_file: str, data_dir: str, accepted_kws: list):
    """Helper function to create a file(training_list.txt) that contains list 
    of all the file paths to training instances.

    Parameters
    ----------
    validation_file: File
        Validation file provided in the dataset.
    testing_file: File
        Test file provided in the dataset.
    data_dir: str
        Path to dataset.
    accepted_kws: list[str]
        List of accepted keywords.

    Returns
    -------
    outputdir: str
        Returns the path where the above file is saved   
    """
    outputdir = data_dir + 'training_list.txt'

    if os.path.exists(outputdir):
        print('There is already a training_list.txt file, in', data_dir, 'aborting!')
        return outputdir

    validation_content = get_file_content_as_list(data_dir + validation_file)
    test_content = get_file_content_as_list(data_dir + testing_file)
    wav_files_paths = [fspath(path) for path in Path(data_dir).rglob('*.wav')]
    train_element_names = []

    for _, wav_path in enumerate(wav_files_paths):
        lab = wav_path.split('/')[-2]
        filename = lab + '/' + wav_path.split('/')[-1]

        if filename not in validation_content and filename not in test_content and lab in accepted_kws:
            train_element_names.append(filename)

    f = open(outputdir, "w")
    for train_element in train_element_names:
        f.write(train_element + '\n')
    f.close()

    return outputdir


def get_file_content_as_list(filename: str):
    """Helper function to return file contents as a list

    Parameters
    ----------
    filename: str
        Name of the file.

    Returns
    -------
    file_contents_list: list[str]
        List of file contents.
    """
    my_file = open(filename, "r")
    file_contents = my_file.read()
    file_contents_list = file_contents.split("\n")
    my_file.close()
    return file_contents_list


def balance_silence_and_unknown(df_all: pd.DataFrame, save_to_path: str = ''):
    """Helper function that returns a version of the dataset containing balanced number of
    silence/unknown utterances per client. It adds more silence and unknown utterances 
    proportionally because it wants to have 10% + 1 utterance on each client.

    Parameters
    ----------
    df_all: pd.DataFrame
        Dataframe consisting of known, unknown and silence utterances.
    save_to_path: str
        Path to save the balanced dataset, defined in `config.py`.

    Returns
    -------
    balanced_df: pd.DataFrame
        Dataframe with balanced number of silence and unknown utterances for FL.
    """
    # Find all silence utterances 
    df_silence = df_all[df_all['keyword'] == 'silence']

    # Get the silence utterances file path
    silence_path = df_silence['file_path'].unique()
    assert len(silence_path) == 1, 'non unique silence paths expected'
    silence_path = silence_path[0]

    # Get one-hot encoding of silence utterances
    silence_one_hot_encoding = df_silence['label_one_hot'].iloc[0]

    # Drop out silence labels
    df_all = df_all[df_all.keyword != 'silence']

    # Balance unknowns in the dataset
    df_all = add_unknown_to_each_speaker(df_all)
    df_all['speaker_id'].nunique()

    # Drop the original silence entries and add the per client utterances
    balanced_df = add_silence_to_each_speaker(df_all[df_all.keyword != 'silence'], silence_path=silence_path,
                                              silence_one_hot_label=silence_one_hot_encoding)

    if save_to_path != '':
        balanced_df.to_csv(save_to_path)
        print('Written dataframe ready for federated learning is saved to ', save_to_path)

    return balanced_df


def add_unknown_to_each_speaker(df: pd.DataFrame, id_col: str = 'speaker_id'):
    """Helper function to add unknown utterances for each speaker in the dataframe

    Parameters
    ----------
    df: pd.DataFrame
        Input dataframe
    id_col: str, optional
        Variable to filter speakers by their IDs. Defaults to `speaker_id`.

    Returns
    -------
    Dataframe with added unknown utterances.
    """
    result = []
    for s in df[id_col].unique():
        speaker_is_s = df[id_col] == s
        df_per_speaker = df[speaker_is_s]

        # Shuffle the unknown keywords from this speaker
        unknown = df_per_speaker[df_per_speaker['keyword'] == 'unknown'].sample(frac=1.)

        # Using keywords not in unknown category
        dfa = df_per_speaker[df_per_speaker.keyword != 'unknown']
        n = int(np.rint(dfa.shape[0] / 10.))

        # Append as many we can as long as they are not more than the percentage of total kws per speaker
        if len(unknown) >= n:
            df_per_speaker = pd.concat([dfa, unknown[0:n]])
        else:
            df_per_speaker = pd.concat([dfa, unknown])

        result.append(df_per_speaker)

    return pd.concat(result)


def add_silence_to_each_speaker(df: pd.DataFrame, silence_path: str, silence_one_hot_label: float,
                                id_col: str = 'speaker_id'):
    """Helper function that adds a fixed percentage(1/11) of silence entries to dataframe, 
    based on its id_col column.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe with file_path, keyword, dataset, speaker_id, speaker_utt.
    silence_path: Any
        File path of silence utterances.
    silence_one_hot_label: Any
        One-hot encoding of silence utterances.
    id_col: str, optional
        Variable to filter speakers by their IDs. Defaults to `speaker_id`.
    
    Returns
    -------
    Dataframe with added silence utterances.

    Raises
    ------
    AssertionError
        If length of `dataset_` equals to `1`.
    """
    result = []
    for speaker in tqdm.tqdm(df[id_col].unique()):
        df_per_speaker = df[df[id_col] == speaker]
        # n_recordings = len(dff)
        add_silence_percent = int(np.rint(df_per_speaker.shape[0] / 11.))
        start_at_id = df_per_speaker['speaker_ut'].apply(lambda x: int(x)).max() + 1
        dataset_ = df_per_speaker['dataset'].unique()
        assert len(dataset_) == 1, 'Speaker not splitted correctly!'
        dataset_ = dataset_[0]

        # Add at least one silence utterance
        for n in range(add_silence_percent):
            result.append(
                {'file_path': silence_path, 'keyword': 'silence', 'dataset': dataset_,
                 'speaker_id': speaker, 'speaker_ut': start_at_id + n}
            )

    result = pd.DataFrame(result)
    result['label_one_hot'] = [silence_one_hot_label] * len(result)
    return df.append(result)


def is_in_basket(speaker_id, basket, n):
    """Function that returns the basket in which each row in dataframe belongs
    """
    for n in range(n):
        if speaker_id in basket[n]:
            return n
    return -1


def partition_fl_set(numbers: dict, num_parts: int):
    """Helper function to partition numbers into num_parts partitions using the 
    kramarkar karp algorithm.

    Parameters
    ----------
    numbers: dict
        Provides cardinality given all the ids.
    num_parts: int
        Number of federated silos to split set into.

    Returns
    -------
    sets: dict
        num_parts list entries that are the client ids.
    cardinality: dict    
        num_parts list entries that are the cardinalities.

    Raises
    ------
    AssertionError
        If the length of `ids` equals to length of `cardinalities`.
    """
    cardinalities = list(numbers.values())
    ids = list(numbers.keys())
    assert len(ids) == len(cardinalities)

    result = karmarkar_karp(cardinalities, num_parts=num_parts, return_indices=True)

    sets = {}
    cardinality = {}
    sums = {}

    for i in range(num_parts):
        # Get original indices
        sets[i] = [ids[j] for j in result.partition[i]]
        # Lookup cardinality in original dict
        cardinality[i] = [numbers[j] for j in sets[i]]
        sums[i] = np.sum(cardinality[i])

    return sets, cardinality


def create_n_roughly_equal_partitions(n: int, df: pd.DataFrame, original_id_column: str = 'speaker_id',
                                      alg: str = 'random', id_name: str = 'fl_id'):
    """Helper function to partition client ids to N roughly equally sized clients

    Parameters
    ----------
    n: int
        Number of resulting unique clients.
    df: pd.DataFrame
        Dataframe with client_ids.
    original_id_column: str, optional
        Column of dataframe used to split dataframe. Defaults to `speaker_id`.
    alg: str, optional
        Name of partitioning algorithm. Defaults to `random`. optionally use 'kk'.
    id_name: str, optional
        Name of the newly generated id. Defaults to `fl_id`.

    Returns
    -------
    df: pd.DataFrame
        Same dataframe with the new id column that contains N freshly generated client ids.
    fl_id_names: Any
        Name of new column `fl_id`.

    Raises
    ------
    AssertionError
        1. If `N` less than number of unique values for `original_id_column` column in df.
        2. If length of `a` greater than `N`.
        3. False, if `alg`(random or kk) is not specified.
    """
    print('Creating ', n, ' roughly equal size partitions')

    # -1 means, the original ids are used
    if n == -1:
        orig_id_to_int = {iid: i for i, iid in enumerate(df[original_id_column].unique())}
        df[id_name] = df[original_id_column].apply(lambda x: orig_id_to_int[x])
        return df, df[original_id_column].unique()

    assert n < df[original_id_column].nunique(), 'num_clients is too big for the # of available ids' + str(
        df[original_id_column].nunique()) + ' N: ' + str(n)
    ids = df[original_id_column].unique()
    id_counts = {curr_id: df[df[original_id_column] == curr_id].shape[0] for curr_id in ids}

    # Partitiion based on the algorithms
    if alg == 'random':
        df['temp_id'] = [i for i in range(df.shape[0])]
        basket = []

        # For every keyword, do equal size splits
        for keyword in df['keyword'].unique():
            filt = df['keyword'] == keyword
            a = df[filt].sample(df[filt].shape[0], random_state=9)['temp_id'].values
            np.random.seed(43)
            assert len(a) > n
            np.random.shuffle(a)
            b = np.array_split(a, n)
            bas = {idd: silo for silo, ids in enumerate(b) for idd in ids}
            basket.append(bas)

        result = {}
        for d in basket:
            for di in d.keys():
                result[di] = d[di]

        df[id_name] = df['temp_id'].apply(lambda x: result[x])
        df = df.drop('temp_id', axis=1)
        fl_id_names = df[df[id_name] != -1][id_name].unique()
        return df, fl_id_names

    elif alg == 'kk':
        basket, _ = partition_fl_set(numbers=id_counts, num_parts=n)

    else:
        assert False, 'Algorithm not implemented'

    df[id_name] = df[original_id_column].apply(lambda x: is_in_basket(x, basket, n))
    fl_id_names = df[df[id_name] != -1][id_name].unique()

    return df, fl_id_names


def gsc_to_dataframe_without_audios(data_dir: str = './', save_to_path: str = ''):
    """Helper function to create labels and metadata with a silence file and silence utterances. 
    Combine all these infos into a dataframe that is saved into the top level of the data_dir.

    Parameters
    ----------
    data_dir: str
        Path to main working directory where the dataset is downloaded. This variable is 
        defined in `config.py`.
        List of keyword labels.
    save_to_path: str
        Where to save the resulting dataframe to. '' means do not save at all. 

    Returns
    -------
    Dataframe with balanced number of silence and unknown utterances for FL.

    Args:
        save_to_path: path to save the resulting dataframe to.
    """
    # Creating a training list file from the validation/testing files provided in the dataset
    kw = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go',
          'unknown']
    create_training_list('validation_list.txt', 'testing_list.txt', data_dir,
                         accepted_kws=['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
                                       'forward',
                                       'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off',
                                       'on', 'right', 'seven',
                                       'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes',
                                       'zero', 'one'])

    kws_all = kw + ['silence']

    # Prepare the metadata information contained in the filenames into a pandas dataframe
    df_all = []
    train_counter = 0

    for se in ['testing', 'validation', 'training']:
        print('Started processing ', se, 'data')
        list_path = data_dir + se + "_list.txt"

        # Read the list of filenames in split
        df = pd.read_csv(list_path, header=None)

        # Extract keyword from filename
        df['keyword'] = df[0].apply(lambda x: x.split('/')[0])

        # Label the unwanted kws as 'unknown'
        df['keyword'] = df.keyword.apply(lambda x: x if x in kw else 'unknown')

        # Split them into known and unknown keywords
        df_unknown = df[df.keyword == 'unknown']
        df = df[df.keyword != 'unknown']

        # Shuffle unknowns
        df_unknown = df_unknown.sample(n=df.shape[0])

        # Add 10% silence utterances. Exact number is not important here, will be rebalanced later
        silence_percentage = 0.1
        df_silence = pd.DataFrame()
        df_silence[0] = ['silence/silence.wav'] * int(silence_percentage * len(df))
        df_silence['keyword'] = ['silence'] * int(silence_percentage * len(df))

        # Combine all into one  df and shuffle
        df = pd.concat([df, df_unknown, df_silence])
        df = df.sample(frac=1, random_state=0)

        # Add dataset info
        if se == 'training':
            df['dataset'] = len(df) * [se]
            train_counter += 1
        else:
            df['dataset'] = len(df) * [se]

        # Add metadata as columns
        df['speaker_info'] = df[0].apply(lambda x: x.split('/')[-1].split('.')[0] if 'silence' not in x else 'None')
        df['speaker_id'] = df['speaker_info'].apply(lambda x: x.split('_')[0] if 'silence' not in x else 'None')
        df['speaker_ut'] = df['speaker_info'].apply(lambda x: x.split('_')[-1] if 'silence' not in x else 'None')
        df['label'] = df['keyword'].apply(lambda x: np.where(np.array(kws_all) == x)[0][0])
        df['speaker_id'] = df.apply(lambda x: x['speaker_id'] if x['keyword'] != 'silence' else
        df.speaker_id.unique()[np.random.randint(len(df.speaker_id.unique()), size=1)[0]], axis=1)
        df['label_one_hot'] = df.label.apply(lambda x: one_at(x, len(kws_all)))
        df = df.drop(['speaker_info', 'label'], axis=1)
        df_all.append(df)

    df_all = pd.concat(df_all)

    # Give unique index to each row
    df_all['idx'] = [i for i in range(len(df_all))]
    df_all = df_all.set_index('idx')

    # Rename columns
    df_all = df_all.rename(columns={0: "file_path"})

    return balance_silence_and_unknown(df_all, save_to_path)


def partition_and_save_dataset(df: pd.DataFrame, n_silos: list = None, part_alg: str = 'random', save_to_path: str = './df_gsc_fl.csv'):
    """Helper function to partition the instances in df into N as n_silos partitions, using either
    kk or random algorithm.

    Parameters
    ----------
    df: pd.Dataframe
        Input preprocessed dataframe.
    n_silos: list[int]
        Desired no of clients/silos/partitions, defined in `config.py`.
    part_alg: str, optional
        Partitioning algorithm(kk or random), else defaults to `random`.
    save_to_path: str, optional
        save resulting dataframe to this path.

    Returns
    -------
    df: pd.Dataframe
        Partitioned dataframe. It is also saved to the directory.
    """
    # Read the preprocessed dataframe
    if n_silos is None:
        n_silos = [2]
    df_train = df[df['dataset'] == 'training']
    df_rest = df[df['dataset'] != 'training']

    # Iterate over all silo splits we want to make
    for num_clients in n_silos:
        curr_name = 'n' + str(num_clients) + '_' + part_alg
        df_train, ids = create_n_roughly_equal_partitions(n=num_clients, df=df_train, original_id_column='speaker_id',
                                                          alg=part_alg,
                                                          id_name=curr_name)
    df = pd.concat([df_train, df_rest])
    df.to_csv(save_to_path)
    return df
