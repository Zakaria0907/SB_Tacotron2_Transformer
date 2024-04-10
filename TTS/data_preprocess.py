import gdown
import tarfile
import os
from tqdm import tqdm
import json
import torchaudio
import csv
import os
import random


def download_with_progress(url, output_path):
    gdown.download(url=url, output=output_path, quiet=False)


def extract_with_progress(tar_path, extract_path):
    with tarfile.open(tar_path, 'r:bz2') as tar:
        members = tar.getmembers()
        total = len(members)

        with tqdm(total=total, desc="Extracting") as pbar:
            for member in members:
                tar.extract(member, path=extract_path)
                pbar.update(1)


def download_and_extract(dataset_url, base_path):
    data_path = os.path.join(base_path, 'data')
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    extraction_check_path = os.path.join(data_path, 'LJSpeech-1.1', 'wavs')  # Path to check if extraction is needed
    if os.path.exists(extraction_check_path):
        print("Dataset already extracted. Skipping extraction.")
        return

    destination_path = os.path.join(data_path, 'LJSpeech-1.1.tar.bz2')

    download_with_progress(dataset_url, destination_path)
    extract_with_progress(destination_path, data_path)

    os.remove(destination_path)
    print("Download and extraction complete.")


def create_json_splits(csv_file, data_folder, split_ratios):
    # Split ratios for train, valid, test
    train_ratio, valid_ratio, _ = split_ratios  # The test ratio is implied

    # Read the CSV and shuffle the rows
    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = list(csv.reader(csvfile, delimiter='|'))
        random.shuffle(reader)  # Shuffle the data

    # Calculate split sizes
    total_size = len(reader)
    train_size = int(total_size * train_ratio)
    valid_size = int(total_size * valid_ratio)

    # Split the data
    train_data = reader[:train_size]
    valid_data = reader[train_size:train_size + valid_size]
    test_data = reader[train_size + valid_size:]
    data_folder_path = 'D:\\PycharmProjects\\SB_Tacotron2_Transformer\\data'

    # Process and save each split
    for split_name, split_data in zip(['train', 'valid', 'test'], [train_data, valid_data, test_data]):
        json_path = os.path.join(data_folder_path, f"{split_name}.json")
        process_and_save_split(split_data, data_folder, json_path)


def process_and_save_split(data, data_folder, json_filename):
    data_dict = {}
    for row in data:
        unique_id = row[0]
        audio_path = os.path.join(data_folder, 'wavs', f"{unique_id}.wav")

        # Load the audio to get its duration
        waveform, sample_rate = torchaudio.load(audio_path)
        duration = waveform.shape[1] / sample_rate

        transcription = row[2].upper() if len(row) > 2 else row[1].upper()

        data_dict[unique_id] = {
            "path": audio_path,
            "duration": duration,
            "normalized_transcription": transcription
        }

    with open(json_filename, 'w') as json_out:
        json.dump(data_dict, json_out, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # Base path where the `data` folder will be located
    path = 'D:\\PycharmProjects\\SB_Tacotron2_Transformer'

    # Google Drive shareable link or direct ID for LJSpeech-1.1.tar.bz2
    url = 'https://drive.google.com/uc?id=1uGZG1UYNtFVlismhjCTi5tpqoHUqdNe0'

    # Download and extract dataset
    download_and_extract(url, path)

    base_folder = 'D:\\PycharmProjects\\SB_Tacotron2_Transformer\\data\\LJSpeech-1.1'
    csv_file_path = os.path.join(base_folder, 'metadata.csv')

    # Split ratios for train, validation, and test sets
    split_ratios = (0.8, 0.1, 0.1)  # For example, 80% train, 10% validation, 10% test

    create_json_splits(csv_file_path, base_folder, split_ratios)
