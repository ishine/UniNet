import random,os
import numpy as np
import torch
import torch.utils.data

import layers
from utils import load_wav_to_torch, load_fbs_and_fb_text_dict, get_alignment_from_times
#from text import text_to_sequence
from hparams import create_hparams
from torch.utils.data import DataLoader
from symbols import phone2id, tone2id, RPB2id

class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, lstfile, hparams):
        self.fbs, self.fb_text_dict = load_fbs_and_fb_text_dict(
            lstfile, hparams.lab_path)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.audio_path = hparams.audio_path
        self.mel_path = hparams.mel_path
        self.MelStd_mel = hparams.MelStd_mel
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(1234)
        random.shuffle(self.fbs)

    def get_mel_text_pair(self, fb):
        # separate filename and text
        text = self.get_text(fb)
        mel = self.get_mel(fb)
        return text, mel

    def get_mel(self, fb):
        if self.load_mel_from_disk:
            cur_mel_path = os.path.join(self.mel_path, fb+'.npy')
            melspec = np.load(cur_mel_path)
            mean, std = np.load(self.MelStd_mel)
            melspec = (melspec - mean) / std
            melspec = np.transpose(melspec)
            melspec = torch.from_numpy(melspec)
        else:
            cur_audio_path = os.path.join(self.audio_path, fb+'.wav')
            audio = load_wav_to_torch(cur_audio_path, self.sampling_rate)
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)  # [mel_bin, T]

            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))
        return melspec

    def get_text(self, fb):
        text_norm = self.fb_text_dict[fb]
        text_norm = torch.LongTensor([[phone2id[ph], tone2id[tn], RPB2id[RPB], int(time)] for ph, tn, RPB, time in text_norm])
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.fbs[index])

    def __len__(self):
        return len(self.fbs)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len, 4) # phone, tone, RPB, time
        text_alignment_padded = torch.LongTensor(len(batch), max_input_len).zero_() - 1 # 全为-1是考虑到NLLLoss可以设置忽略的索引 相当于mask
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            pos = ids_sorted_decreasing[i]
            text = batch[pos][0]
            text_padded[i, :text.size(0), :] = text
            for j in range(text.size(0)):
                text_alignment_padded[i, j] = j

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0) #80
        max_target_len = max([x[1].size(1) for x in batch]) 
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        alignments_padded = torch.FloatTensor(len(batch), max_target_len, max_input_len + 1)
        alignments_padded.zero_()
        alignments_weights_padded = torch.ones(len(batch), max_target_len).float()

        for i in range(len(ids_sorted_decreasing)):
            pos = ids_sorted_decreasing[i]
            mel = batch[pos][1]
            mel_padded[i, :, :mel.size(1)] = mel
            output_lengths[i] = mel.size(1)
            times = text_padded[i,:,3].squeeze()
            alignments_padded[i,:,:] = torch.from_numpy(get_alignment_from_times(times,
                                                          shape=(max_input_len + 1, max_target_len)))
            end_time = np.unique(np.insert(np.cumsum(times), 0, 0)[1:] - 1)
            alignments_weights_padded[i, end_time] = 3

        return text_padded, text_alignment_padded, input_lengths, mel_padded, alignments_padded, alignments_weights_padded,\
            output_lengths

if __name__ == "__main__":
    from hparams import create_hparams
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from model import Tacotron2
    from loss_function import Tacotron2Loss
    hparams = create_hparams()
    text_loader = TextMelLoader(hparams.training_lst, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    text, mel = text_loader[0] # mel.shape (80 * frame_num)
    plt.matshow(mel, origin='lower')
    plt.colorbar()
    plt.savefig('mel_demo.png')
    
    train_loader = torch.utils.data.DataLoader(
                            text_loader, 
                            num_workers=1,  shuffle=False,
                            batch_size=3,  pin_memory=False,
                            drop_last=True, collate_fn=collate_fn)
    print(len(train_loader))
    tacotron = Tacotron2(hparams)
    criterion = Tacotron2Loss()
    for batch in train_loader:
        text_padded, text_alignment_padded, input_lengths, mel_padded, alignments, alignments_weights_padded,\
            output_lengths = batch
        max_len = torch.max(input_lengths.data).item()
        x = (text_padded, input_lengths, mel_padded, max_len, output_lengths)
        y = (mel_padded, alignments, alignments_weights_padded, text_alignment_padded)
        y_pred = tacotron(x)
        print(criterion(y_pred, y))
        break