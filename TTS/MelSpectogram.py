import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import librosa


class MelSpectrogram:
    def __init__(self, sample_rate=22050, n_fft=2048, win_length=None, hop_length=512, n_mels=128):
        """
        Initializes the MelSpectrogram transformer.

        :param sample_rate: Sample rate of the audio files.
        :param n_fft: Number of FFT components.
        :param win_length: Each frame of audio is windowed by window of length win_length and then padded with zeros to match n_fft.
        :param hop_length: Number of audio samples between adjacent STFT columns.
        :param n_mels: Number of Mel bands to generate.
        """
        self.mel_spectrogram_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            n_mels=n_mels,
            mel_scale="htk",
        )

    def __call__(self, waveform):
        """
        Applies the MelSpectrogram transformation to an audio waveform.

        :param waveform: A tensor of audio samples.
        :returns: Mel spectrogram tensor.
        """
        return self.mel_spectrogram_transform(waveform)

    def plot_spectrogram(self, specgram, title=None, ylabel="freq_bin", ax=None):

        fig, _ = plt.subplots(1, 1, figsize=(15, 8))
        if ax is None:
            _, ax = plt.subplots(1, 1)
        if title is not None:
            ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")
        fig.tight_layout()

        plt.show()


# Usage example
if __name__ == "__main__":
    # Load audio
    SAMPLE_SPEECH = torchaudio.utils.download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
    SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(SAMPLE_SPEECH)

    # Initialize MelSpectrogram transformer
    mel_spectrogram = MelSpectrogram(sample_rate=SAMPLE_RATE)

    # Apply transformer to get Mel spectrogram
    mel = mel_spectrogram(SPEECH_WAVEFORM)

    print(f"Mel Spectrogram Shape: {mel.shape}")
    print(f"Mel Spectrogram: {mel[0]}")
    print(f"Type Spectrogram Shape: {type(mel)}")

    mel_spectrogram.plot_spectrogram(mel[0], title="MelSpectrogram - torchaudio", ylabel="mel freq")
