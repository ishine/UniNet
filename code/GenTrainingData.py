import sys,os,re
sys.path.append('..')

import layers
from hparams import create_hparams
from glob import glob
from utils import load_wav_to_torch
import torch
import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def SaveMkdir(dir):
    try:
        if not os.path.exists(dir):
            os.mkdir(dir)
    except:
        os.makedirs(dir)

# 针对目录计算均值数据
def cal_MeanStd(datadir, dim, ref_file=None):
    # This method is efficient for large datadir
    # First row is mean vector
    # Second row is std vector
    tqdm.write('Calculate MeanStd Mean File...')
    files = os.listdir(datadir) 
    if ref_file!=None: 
        ref_list = np.loadtxt(ref_file,'str')
        files = [file for file in files if file.split('.')[0] in ref_list]
    filenum = len(files)
    mean_std = np.zeros([2,dim],dtype=np.float64)
    file_mean = np.zeros([filenum,dim+1],dtype=np.float64)
    file_std = np.zeros([filenum,dim+1],dtype=np.float64)
    for i in tqdm(range(len(files))):     
        file = datadir+os.sep+files[i]
        data = np.load(file)
        file_mean[i][0] = data.shape[0]
        file_std[i][0] = data.shape[0]      
        file_mean[i][1:] = np.mean(data,0)
        file_std[i][1:] = np.mean(data**2,0)
    
    file_sum = (file_mean[:,0]*file_mean[:,1:].T).T 
    file_ssum = (file_std[:,0]*file_std[:,1:].T).T 
    
    mean_std[0] = np.sum(file_sum,0) / np.sum(file_mean[:,0])   
    mean_std[1] = np.sqrt(np.sum(file_ssum,0)/ np.sum(file_mean[:,0]) - mean_std[0]**2)
    return mean_std

def text_to_sequence(absolute_path, mel_path):
    # 将cutedlab转为音素+音调的形式
    if mel_path != None:
        mel_total_time = np.load(mel_path).shape[0]
    phones_num = int(len(open(absolute_path).readlines())/5)
    lis = []
    time_lis = []
    with open(absolute_path, 'rt') as fp:
        for i in range(phones_num):
            for j in  range(5):
                line = fp.readline().strip().split()
                if j == 0:
                    start = int(line[0])
                if j == 4:
                    end = int(line[1])
            # 传统是5ms一帧 现在hop_size是240个点一帧(15ms) 所以需要除以3
            time = (end-start)/(50000*3)
            p = re.sub(extract_phoneme,'\\1',line[2])
            t = re.sub(extract_tone,'\\1',line[2])
            r = re.sub(extract_Rhythm_phrase_boundary,'\\1',line[2])
            if p == 'sil':
                t = r = 'sil'
            lis.append((p,t,r))
            time_lis.append(time)
            #lis.append('%-7s %-7s %-7s %-7d'%(p,t,r,time))
    if mel_path != None:
        total_time = np.sum(time_lis)
        time_lis = time_lis/total_time*mel_total_time
    processed_time_lis = getIntDurInfo(np.array(time_lis, dtype=np.float64))
    
    result = []
    for i in range(len(lis)):
        result.append('%-7s %-7s %-7s %-7d'%(lis[i][0],lis[i][1],lis[i][2], processed_time_lis[i]))
    return result

# 根据音素时长动态调整状态时长
def getIntDurInfo(lis):
    lis_acc=np.cumsum(lis)
    result=np.zeros(len(lis))
    result[0]=np.round(lis[0])
    if(result[0]<=0):
        result[0]=1 #保证状态时长不为零
    for i in range(1,len(lis)):
        result[i]=np.round(lis_acc[i]-np.round(np.sum(result)))
        if(result[i]<=0):
            result[i]=1 #保证状态时长不为零
    return result

if __name__ == "__main__":
    gen_mel = 0
    gen_mel_png = 0
    gen_text = 0
    check_len = 0
    #测试集的时长来自于DNN的预测
    Gen_test_text_addtime = 1

    if gen_mel:
        hparams = create_hparams()
        stft = layers.TacotronSTFT(
                hparams.filter_length, hparams.hop_length, hparams.win_length,
                hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
                hparams.mel_fmax)

        audio_files = sorted(glob('../data/audio/*.wav'))
        out_dir = '../data/mel_15ms'
        SaveMkdir(out_dir)
        for file in tqdm(audio_files):
            file_basename = os.path.basename(file).split('.')[0]
            tqdm.write(file_basename)
            audio_path = os.path.join(hparams.audio_path, file_basename+'.wav')
            audio = load_wav_to_torch(audio_path, hparams.sampling_rate)
            audio_norm = audio / hparams.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = stft.mel_spectrogram(audio_norm)

            #转置存错 即数据行代表帧 列代表特征
            melspec = torch.squeeze(melspec, 0).numpy().transpose()

            out_file = os.path.join(out_dir, file_basename+'.npy')
            np.save(out_file, melspec)

        mean_std = cal_MeanStd(out_dir,hparams.n_mel_channels, ref_file='../filelists/train+val_file.lst')
        mean_std_file = os.path.join(out_dir, os.pardir, 'MeanStd_Tacotron_mel_15ms.npy')
        np.save(mean_std_file,mean_std)

    if gen_mel_png:
        out_dir = '../data/mel_15ms'
        files = sorted(glob(os.path.join(out_dir, '*.npy')))
        mean_std_file = os.path.join(out_dir, os.pardir, 'MeanStd_Tacotron_mel_15ms.npy')

        for file in tqdm(files):
            file_basename = os.path.basename(file).split('.')[0]
            tqdm.write(file_basename)
            out_file = os.path.join(out_dir, file_basename+'.npy')
            melspec = np.load(out_file)
            mean, std = np.load(mean_std_file)
            melspec = (melspec - mean) / std
            melspec = np.transpose(melspec)

            plt.matshow(melspec, origin='lower')
            plt.colorbar()
            mel_path = os.path.join(out_dir, file_basename+'_normlize.png')
            plt.savefig(mel_path)
            plt.close()

    if gen_text:
        cutedlab = '../data/cutedlab'
        text_dir = '../data/text_addtime'
        mel_dir = '../data/mel_15ms'
        SaveMkdir(text_dir)

        lab_files = sorted(glob(os.path.join(cutedlab, '*.lab')))  
        extract_phoneme = re.compile(r'.*-(.*)\+.*')
        extract_tone = re.compile(r'.*@(.*)\$.*/B.*')
        extract_Rhythm_phrase_boundary = re.compile(r'.*/B:(.)_.*')

        tone_types = set()
        Rhythm_phrase_boundary_types = set()
        for file in tqdm(lab_files):
            file_basename = os.path.basename(file).split('.')[0]
            tqdm.write(file_basename)
            lab_path = os.path.join(cutedlab, file_basename+'.lab')
            text_path = os.path.join(text_dir, file_basename+'.lab')
            mel_path = os.path.join(mel_dir, file_basename+'.npy')
            text = text_to_sequence(lab_path, mel_path)
            np.savetxt(text_path, text, fmt='%s')
            
            tones = map(lambda i:i.split()[1], text)
            tone_types = tone_types | set(tones)

            Rhythm_phrase_boundary = map(lambda i:i.split()[2], text)
            Rhythm_phrase_boundary_types = Rhythm_phrase_boundary_types|set(Rhythm_phrase_boundary)
        print(sorted(list(tone_types)))
        print(sorted(list(Rhythm_phrase_boundary_types)))

    if check_len==1:
        text_dir = '../data/text_addtime'
        mel_dir = '../data/mel_15ms'

        for file in tqdm(sorted(os.listdir(text_dir))):
            file_basename = file.split('.')[0]
            tqdm.write(file_basename)
            text_path = os.path.join(text_dir, file_basename+'.lab')
            mel_path = os.path.join(mel_dir, file_basename+'.npy')

            mel_data = np.load(mel_path)
            text_data = np.loadtxt(text_path, 'str')
            assert(mel_data.shape[0]==np.sum(np.array(text_data[:,-1], dtype=np.float)))


    if Gen_test_text_addtime:
        cutedlab_dir = '../tacotron2_baseline_medium/Linguistic2Dur/gen_test_cutedlab'
        text_dir = '../tacotron2_baseline_medium/Linguistic2Dur/gen_test_text_addtime'

        SaveMkdir(text_dir)
        lab_files = sorted(glob(os.path.join(cutedlab_dir, '*.lab')))
        extract_phoneme = re.compile(r'.*-(.*)\+.*')
        extract_tone = re.compile(r'.*@(.*)\$.*/B.*')
        extract_Rhythm_phrase_boundary = re.compile(r'.*/B:(.)_.*')

        tone_types = set()
        Rhythm_phrase_boundary_types = set()
        for file in tqdm(lab_files):
            file_basename = os.path.basename(file).split('.')[0]
            tqdm.write(file_basename)
            lab_path = os.path.join(cutedlab_dir, file_basename+'.lab')
            text_path = os.path.join(text_dir, file_basename+'.lab')
            text = text_to_sequence(lab_path, mel_path=None)
            np.savetxt(text_path, text, fmt='%s')
            
            tones = map(lambda i:i.split()[1], text)
            tone_types = tone_types | set(tones)

            Rhythm_phrase_boundary = map(lambda i:i.split()[2], text)
            Rhythm_phrase_boundary_types = Rhythm_phrase_boundary_types|set(Rhythm_phrase_boundary)
        print(sorted(list(tone_types)))
        print(sorted(list(Rhythm_phrase_boundary_types)))