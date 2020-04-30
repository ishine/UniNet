import numpy as np
from scipy.io.wavfile import read
import torch,os


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    #ids = torch.arange(0, max_len, out=torch.LongTensor(max_len)) #CPU version
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).byte()
    return mask


def get_alignment_from_times(times, shape):
    times_accu = np.insert(np.cumsum(times), 0, 0, axis=0)
    start_idx = times_accu[:-1]
    end_idx = times_accu[1:]
    mat = np.zeros(shape)
    l = np.count_nonzero(times)
    for i in range(l):
        mat[i, start_idx[i]:end_idx[i] - 1] = 1
        mat[i + 1, end_idx[i] - 1] = 1
    return mat.T

# 更易于理解的get_alignment_from_times代码
'''
import numpy as np
def get_alignment_from_times(times, shape):
    times_accu = np.insert(np.cumsum(times), 0, 0, axis=0)
    start_idx = times_accu[:-1]
    end_idx = times_accu[1:]
    mat = np.zeros(shape)
    for i in range(np.count_nonzero(times)):
        mat[i, start_idx[i]:end_idx[i] - 1] = 1
        mat[i+1, end_idx[i]-1] = 1
    mat = np.flipud(mat)
    return mat

a=get_alignment_from_times([4,3,1,0,0,0], shape=(7,8))
print(a)
'''

def get_mask_from_times(phone_idx, lengths):
    mask = torch.ones(lengths, dtype=torch.uint8)
    mask[phone_idx:phone_idx+2] = 0
    return mask


def get_start_end_times(times):
    phone_time_accu = np.insert(np.cumsum(times, axis=1), 0, 0, axis=1)
    phone_start_idx = phone_time_accu[:,:-1]
    phone_end_idx = phone_time_accu[:,1:] - 1
    # fix start time shift problem     
    #for i in range(len(phone_time_accu)):
    #    error_idx = np.where(phone_start_idx[i] >= phone_end_idx[i])[0]
    #    if len(error_idx) > 0:
    #        phone_start_idx[i, error_idx] = phone_start_idx[i, error_idx[0] - 1]
    return phone_start_idx, phone_end_idx


def load_wav_to_torch(full_path, sr):
    sampling_rate, data = read(full_path)
    assert sr == sampling_rate, "{} SR doesn't match {} on path {}".format(
        sr, sampling_rate, full_path)
    return torch.FloatTensor(data.astype(np.float32))


def load_fbs_and_fb_text_dict(filename, text_path_root):
    fbs = np.loadtxt(filename, 'str')
    fb_text_dict = {}
    for fb in fbs:
        text_path = os.path.join(text_path_root, fb+'.lab')
        text = np.loadtxt(text_path, 'str')
        fb_text_dict[fb] = text
    return fbs, fb_text_dict


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)
