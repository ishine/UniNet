import sys
sys.path.append('../tools')

from tools import *
from functions import *
import numpy as np
import torch,librosa,os
from tqdm import tqdm
import soundfile as sf
import scipy.io as sio
from scipy.special import softmax

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from symbols import phone2id, tone2id, RPB2id

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cbook import flatten
import pdb
from time import time

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

def recover_wav(mel):
    n_fft = 2048
    win_length=hparams.win_length
    hop_length=hparams.hop_length
    mean, std = np.load( "../data/MeanStd_Tacotron_mel_15ms.npy")
    mel = mel.transpose()
    temp1=np.tile(mean, (mel.shape[0],1))
    temp2=np.tile(std, (mel.shape[0],1))
    mel = mel * std + mean
    mel = np.exp(mel).transpose()
    filters = librosa.filters.mel(sr=hparams.sampling_rate, n_fft=n_fft, n_mels=hparams.n_mel_channels)
    inv_filters = np.linalg.pinv(filters)
    spec = np.dot(inv_filters, mel)

    def _griffin_lim(stftm_matrix, shape, max_iter=50):
        y = np.random.random(shape)
        for i in range(max_iter):
            stft_matrix = librosa.core.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
            stft_matrix = stftm_matrix * stft_matrix / np.abs(stft_matrix)
            y = librosa.core.istft(stft_matrix, win_length=win_length, hop_length=hop_length)
        return y

    shape = spec.shape[1] * hop_length -  hop_length + 1

    y = _griffin_lim(spec, shape)
    return y

def SaveMkdir(dir):
    try:
        if not os.path.exists(dir):
            os.mkdir(dir)
    except:
        os.makedirs(dir)

if __name__ == "__main__":
    GenAudio = 0
    Cal_Distortion_wav = 0
    Cal_Distortion_mel = 0
    GenAudio_BC2019 = 1
    force_alignment = 0

    if GenAudio:
        hparams = create_hparams()

        checkpoint_path = '../outdir/tacotron2_baseline_medium_modified_v4_0.03_j3/checkpoint_28980'
        model = load_model(hparams)
        model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
        _ = model.cuda().eval()

        test_text_dir = '../data/text'
        audio_dir = '../outdir/tacotron2_baseline_medium_modified_v4_0.03_j3/GenAudio0.5'
        test_list = np.loadtxt('../filelists/test_file.lst', 'str')

        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of parameter: %.2fM, Number of trainable parameter: %.2fM" % (total_num / 1e6, trainable_num/1e6))

        SaveMkdir(audio_dir)
        p_time_list = []
        total_time = totol_frame_len = 0
        for fb in tqdm(test_list):
            tqdm.write(fb)
            text_path = os.path.join(test_text_dir, fb+'.lab')
            audio_path = os.path.join(audio_dir, fb+'.wav')
            alignments_path = os.path.join(audio_dir, fb+'.png')
            alignments_text_path = os.path.join(audio_dir, fb+'_text.png')
            mel_outputs_postnet_path = os.path.join(audio_dir, fb+'_mel.png')
            mel_data_path = os.path.join(audio_dir, fb+'.dat')
            duration_path = os.path.join(audio_dir, fb+'.txt')

            text = np.loadtxt(text_path, 'str')
            sequence = np.array([[phone2id[ph], tone2id[tn], RPB2id[RPB]] for ph, tn, RPB in text])[None, :]
            sequence = torch.autograd.Variable(
                torch.from_numpy(sequence)).cuda().long()

            start = time()
            mel_outputs, mel_outputs_postnet, alignments = model.inference(sequence)
            total_time = total_time + time() - start
            totol_frame_len = totol_frame_len + mel_outputs.shape[2]
            mel_outputs_postnet = mel_outputs_postnet[0].detach().cpu().numpy()
            mel_outputs_postnet.astype(np.float32).transpose().tofile(mel_data_path)

            ### For paper
            alignments_hard = (alignments > 0.5).float() #For paper
            temp_one_hot = np.zeros(alignments_hard.shape[2])
            temp_one_hot[0] = 1
            temp_one_hot = temp_one_hot.reshape(1,1,-1)
            alignments_hard = np.concatenate((temp_one_hot,alignments_hard.detach().cpu().numpy()[:,:-1,:]),axis=1)
            alignments_hard = alignments_hard[0,:,:-1].T 
            ### For paper

            alignments = alignments[0].detach().cpu().numpy().T
            p_time=np.asarray((np.unique(alignments.argmax(axis=0), return_counts=True)))[1]
            p_time[0]=p_time[0] + 1
            p_time = p_time[:-1] * 3
            p_time_list.append(p_time)
            np.savetxt(duration_path, np.c_[text[:, 0], p_time], '%s')

            #sio.savemat(alignments_path, {'data':alignments})
            pltfig = plt.figure(figsize=(120, 30))
            plt.matshow(alignments_hard, origin='lower', fignum=pltfig.number)
            plt.colorbar()
            plt.savefig(alignments_path)
            plt.close()

            plt.matshow(mel_outputs_postnet, origin='lower')
            plt.colorbar()
            plt.savefig(mel_outputs_postnet_path)
            plt.close()
            
            audio=recover_wav(mel_outputs_postnet)
            audio = librosa.util.normalize(audio, norm=np.inf, axis=None)
            sf.write(audio_path, audio, hparams.sampling_rate, 'PCM_16')
        #sio.savemat(os.path.join(audio_dir, 'mask_pos_dict_all.mat'), mask_pos_dict_all)
        p_time_list = np.concatenate(p_time_list)
        np.savetxt(os.path.join(audio_dir, 'duration.lst'), p_time_list, fmt='%s')
        print("Mean Elapsed time is %.6fs" % (float(total_time)/totol_frame_len))

    if Cal_Distortion_wav:
        wav_gen_dir = '../wavenet_vocoder/egs/mol/test_set/tacotron2_baseline_medium_modified_v4_0.03_j3'
        wav_ref_dir = '../data/audio'
        ref_file='../filelists/test_file.lst'
        Compute_wavs_Distortion(wav_gen_dir, wav_ref_dir, 13, ref_file, 'fastdtw')

    if Cal_Distortion_mel:
        mel_gen_dir = '../outdir/tacotron2_baseline_medium_modified_v4_0.03_j3/GenAudio0.5'
        mel_ref_dir = '../data/mel_15ms'
        ref_file='../filelists/test_file.lst'
        mean_file = '../data/MeanStd_Tacotron_mel_15ms.npy'
        Compute_mels_Distortion(mel_gen_dir, mel_ref_dir, mean_file, 80, ref_file, 'fastdtw')

    if GenAudio_BC2019:
        hparams = create_hparams()

        checkpoint_path = '../outdir/tacotron2_baseline_medium_modified_v4_0.03_j3/checkpoint_28980'
        model = load_model(hparams, dev='cpu')
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['state_dict'])
        _ = model.eval()

        test_text_dir = '../data/BC2019/text'
        audio_dir = '../outdir/tacotron2_baseline_medium_modified_v4_0.03_j3/GenAudio0.5_BC2019_verylong'
        test_list = np.loadtxt('../data/BC2019/test_files_verylong.lst', 'str')

        SaveMkdir(audio_dir)
        for fb in tqdm(test_list):
            tqdm.write(fb)
            text_path = os.path.join(test_text_dir, fb+'.lab')
            audio_path = os.path.join(audio_dir, fb+'.wav')
            #mel_data_path = os.path.join(audio_dir, fb+'.dat')
            #alignments_path = os.path.join(audio_dir, fb+'.png')
            #duration_path = os.path.join(audio_dir, fb+'.txt')

            text = np.loadtxt(text_path, 'str')
            sequence = np.array([[phone2id[ph], tone2id[tn], RPB2id[RPB]] for ph, tn, RPB in text])[None, :]
            sequence = torch.autograd.Variable(
                torch.from_numpy(sequence)).long()

            mel_outputs, mel_outputs_postnet, alignments = model.inference(sequence)

            mel_outputs_postnet = mel_outputs_postnet[0].detach().cpu().numpy()
            #mel_outputs_postnet.transpose().tofile(mel_data_path)
            '''
            alignments = alignments[0].detach().cpu().numpy().T
            p_time=np.asarray((np.unique(alignments.argmax(axis=0), return_counts=True)))[1]
            p_time[0]=p_time[0] + 1
            p_time = p_time[:-1] * 3
            np.savetxt(duration_path, np.c_[text[:, 0], p_time], '%s')

            pltfig = plt.figure(figsize=(120, 30))
            plt.matshow(alignments, origin='lower', fignum=pltfig.number)
            plt.colorbar()
            plt.savefig(alignments_path)
            plt.close()
            '''
            
            audio=recover_wav(mel_outputs_postnet)
            audio = librosa.util.normalize(audio, norm=np.inf, axis=None)
            sf.write(audio_path, audio, hparams.sampling_rate, 'PCM_16')

    if force_alignment:
        hparams = create_hparams()

        checkpoint_path = '../outdir/tacotron2_baseline_medium_modified_v4_0.03_j3/checkpoint_28980'
        model = load_model(hparams)
        model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
        _ = model.cuda().eval()

        assert(hparams.more_information==False)
        # more_information = False
        # 另外还要把prenet的dropout是True
        # 即x = F.dropout(F.relu(linear(x)), p=0.5, training=True)

        '''
        text_dir = '../data/text_addtime'
        mel_path = '../data/mel_15ms'
        mean_std_file = '../data/MeanStd_Tacotron_mel_15ms.npy'
        mean, std = np.load(mean_std_file)

        fb = '00000070'
        test_text_path = os.path.join(text_dir, fb+'.lab')
        sequence = np.loadtxt(test_text_path, 'str')
        sequence = np.array([[phone2id[ph], tone2id[tn], RPB2id[RPB], int(time)] for ph, tn, RPB, time in sequence])[None, :]

        text_padded = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
        input_lengths = torch.LongTensor([text_padded.size(1)]).cuda()

        mel_padded = np.load(os.path.join(mel_path, fb+'.npy'))
        mel_padded = np.transpose((mel_padded - mean) / std)
        np.save('ref_mel.npy', mel_padded)
        mel_padded = torch.FloatTensor(mel_padded[None,:,:]).cuda()

        max_len = torch.max(input_lengths.data).item()
        output_lengths = torch.LongTensor([mel_padded.size(2)]).cuda()
        
        x = (text_padded, input_lengths, mel_padded, max_len, output_lengths)
        mel_outputs, mel_outputs_postnet, alignments, _, _, _ = model(x)
        alignments = alignments[0].detach().cpu().numpy()
        jump = alignments[alignments>0].reshape(-1,2)[:,1]

        sio.savemat('uninet.mat', {'data':jump, 'time':sequence[0,:,3]})
        #pdb.set_trace()
        '''

        test_text_dir = '../data/text'
        fb = '00000070'
        text_path = os.path.join(test_text_dir, fb+'.lab')        
        text = np.loadtxt(text_path, 'str')
        sequence = np.array([[phone2id[ph], tone2id[tn], RPB2id[RPB]] for ph, tn, RPB in text])[None, :]
        sequence = torch.autograd.Variable(
            torch.from_numpy(sequence)).cuda().long()
        mel_outputs, mel_outputs_postnet, alignments = model.inference(sequence)

        alignments = alignments[0].detach().cpu().numpy()
        jump = alignments[alignments>0].reshape(-1,2)[:,1]
        print(alignments)
        p_time=np.asarray((np.unique(alignments.argmax(axis=1), return_counts=True)))[1]
        p_time[0] = p_time[0] + 1
        p_time[-1] = p_time[-1] - 1
        print(p_time, np.sum(p_time))
        sio.savemat('uninet2.mat', {'data':jump, 'time':p_time})
