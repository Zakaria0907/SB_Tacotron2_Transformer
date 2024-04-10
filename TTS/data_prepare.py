import torch
import speechbrain as sb
import sys
import logging
from hyperpyyaml import load_hyperpyyaml

from speechbrain.inference import GraphemeToPhoneme
from speechbrain.utils.text_to_sequence import text_to_sequence, _g2p_keep_punctuations, _clean_text
from speechbrain.utils.data_utils import scalarize
import speechbrain.dataio.dataset
from speechbrain.dataio.dataio import read_audio
import speechbrain.utils.data_pipeline
import torchaudio
import matplotlib.pyplot as plt

from TTS.MelSpectogram import MelSpectrogram


def dataio_prepare(hparams):
    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["train_json"],
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["valid_json"],
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["test_json"],
    )

    datasets = [train_data, valid_data, test_data]

    # Done
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("mel")
    def audio_spec_pipeline(wav):
        mel_spectogram = MelSpectrogram(sample_rate=22050, n_fft=2048, win_length=None, hop_length=512, n_mels=128)
        audio = sb.dataio.dataio.read_audio(wav)
        mel = mel_spectogram(audio)

        return mel

    sb.dataio.dataset.add_dynamic_item(datasets, audio_spec_pipeline)

    # Done
    @sb.utils.data_pipeline.takes("transcript")
    @sb.utils.data_pipeline.provides("text_phonemes", "embedded_phonemes")
    def text_pipeline(transcript):
        g2p = GraphemeToPhoneme.from_hparams("speechbrain/soundchoice-g2p", savedir="pretrained_models/soundchoice-g2p")

        # Clean the transcript
        clean_transcript = _clean_text(transcript, ["english_cleaners"])

        # Convert the transcript to phonemes
        phonemes_list = _g2p_keep_punctuations(g2p, clean_transcript)
        phonemes_str = ''.join(phonemes_list)

        # Convert the text to a sequence and to a tensor
        embedded_phonemes = torch.IntTensor(text_to_sequence(phonemes_str, ["english_cleaners"]))

        yield phonemes_str, embedded_phonemes

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    sb.dataio.dataset.set_output_keys(
        datasets, ["wav", "mel", "transcript", "phonemes_str", "embedded_phonemes"],
    )

    return (
        train_data,
        valid_data,
        test_data,
    )
