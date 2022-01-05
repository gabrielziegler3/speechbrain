"""
Downloads and creates data manifest files for Mini LibriSpeech (spk-id).
For speaker-id, different sentences of the same speaker must appear in train,
validation, and test sets. In this case, these sets are thus derived from
splitting the original training set intothree chunks.

Authors:
 * Mirco Ravanelli, 2021
"""
import re
import os
import json
import shutil
import random
import logging
import torchaudio
from speechbrain.utils.data_utils import get_all_files, download_file
from speechbrain.dataio.dataio import read_audio
from sklearn.model_selection import train_test_split
from coraa_dataset import CORAADataset


logger = logging.getLogger(__name__)
SAMPLERATE = 16000


def prepare_coraa(
    data_folder,
    save_json_train,
    save_json_valid,
    save_json_test,
    split_ratio=[80, 10, 10],
):
    """
    Prepares the json files for the Mini Librispeech dataset.

    Downloads the dataset if it is not found in the `data_folder`.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the Mini Librispeech dataset is stored.
    save_json_train : str
        Path where the train data specification file will be saved.
    save_json_valid : str
        Path where the validation data specification file will be saved.
    save_json_test : str
        Path where the test data specification file will be saved.
    split_ratio: list
        List composed of three integers that sets split ratios for train, valid,
        and test sets, respectively. For instance split_ratio=[80, 10, 10] will
        assign 80% of the sentences to training, 10% for validation, and 10%
        for test.

    Example
    -------
    >>> data_folder = '/path/to/mini_librispeech'
    >>> prepare_mini_librispeech(data_folder, 'train.json', 'valid.json', 'test.json')
    """

    # Check if this phase is already done (if so, skip it)
    if skip(save_json_train, save_json_valid, save_json_test):
        logger.info("Preparation completed in previous run, skipping.")
        return

    # List files and create manifest from list
    logger.info(
        f"Creating {save_json_train}, {save_json_valid}, and {save_json_test}"
    )
    # wav_list = get_all_files(data_folder, match_and=extension)
    coraa_dataset = CORAADataset(data_folder, model_sr=SAMPLERATE)

    # Random split the signal list into train, valid, and test sets.
    train_split, val_split, test_split = split_sets(coraa_dataset.audio_files,
                                                    coraa_dataset.labels)

    # Creating json files
    create_json(train_split[0], save_json_train)
    create_json(val_split[0], save_json_valid)
    create_json(test_split[0], save_json_test)


def create_json(wav_list, json_file):
    """
    Creates the json file given a list of wav files.

    Arguments
    ---------
    wav_list : list of str
        The list of wav files.
    json_file : str
        The path of the output json file
    """
    # Processing all the wav files in the list
    json_dict = {}
    for wav_file in wav_list:

        # Reading the signal (to retrieve duration in seconds)
        signal, _ = torchaudio.load(wav_file)
        duration = signal.shape[1] / SAMPLERATE

        # Manipulate path to get relative path and uttid
        path_parts = wav_file.split(os.path.sep)
        uttid, _ = os.path.splitext(path_parts[-1])
        relative_path = wav_file

        # Getting speaker-id from utterance-id
        spk_id = re.search(r"^([^_]*)", uttid).group(1)
        sentiment = re.search(r"([^_]+)$", uttid).group(1)

        # Create entry for this utterance
        json_dict[uttid] = {
            "wav": relative_path,
            "length": duration,
            "spk_id": spk_id,
            "sentiment": sentiment
        }

    # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created!")


def skip(*filenames):
    """
    Detects if the data preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True


def check_folders(*folders):
    """Returns False if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True


def split_sets(wav_list, labels_list, test_size=0.1, validation_size=0.1):
    X_train, X_val, y_train, y_val = train_test_split(wav_list, labels_list,
                                                      test_size=0.2, stratify=labels_list,
                                                      random_state=42)

    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5,
                                                    stratify=y_val, random_state=42)

    from collections import Counter

    logger.info(f"Train: {len(y_train)} {Counter(y_train)}")
    logger.info(f"Val:   {len(y_val)}   {Counter(y_val)}")
    logger.info(f"Test:  {len(y_test)}  {Counter(y_test)}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

