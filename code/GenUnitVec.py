import numpy as np
import torch,os,librosa
from tqdm import tqdm
import hickle as hkl
from hparams import create_hparams
from model import Tacotron2
from train import load_model
from symbols import phone2id, tone2id, RPB2id
import soundfile as sf
import pdb
import scipy.io as sio

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

def ReadFloatRawMat(datafile,column):
    data = np.fromfile(datafile,dtype=np.float32)
    assert len(data)%column == 0, 'ReadFloatRawMat %s, column wrong!'%datafile
    assert len(data) != 0, 'empty file: %s'%datafile
    data.shape = [int(len(data)/column), column]
    return np.float64(data)

def WriteArrayFloat(file,data):
    tmp=np.array(data,dtype=np.float32)
    tmp.tofile(file)

hparams = create_hparams()
checkpoint_path = "../outdir/tacotron2_baseline_medium_modified_v4_0.03_j3/checkpoint_28980"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval()

text_dir = '../data/text'

A=0
B=0
C=0
extract_info=0

C_text=0
extract_info_test=0

gen_embedding_plot = 0

if A:
    unit_vector = []
    unit_vector_path = '../outdir/tacotron2_baseline_medium_modified_v4_0.03_j3/unit_vector/Unit2Vec_UnitVector.dat'
    train_val_file = '../filelists/train+val_file.lst'
    train_val_file = np.loadtxt(train_val_file, 'str')
    files  = filter(lambda i:i.split('.')[0] in train_val_file, os.listdir(text_dir))

    for fb in tqdm(sorted(files)):
        tqdm.write(fb)
        test_text_path = os.path.join(text_dir, fb)
        sequence = np.loadtxt(test_text_path, 'str')
        sequence = np.array([[phone2id[ph], tone2id[tn], RPB2id[RPB]] for ph, tn, RPB in sequence])[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

        embedded_inputs_phoneme = model.embedding_phoneme(sequence[:,:,0]).transpose(1, 2)
        embedded_inputs_tone = model.embedding_tone(sequence[:,:,1]).transpose(1, 2)
        embedded_inputs_RPB = model.embedding_RPB(sequence[:,:,2]).transpose(1, 2)
        embedded_inputs = torch.cat((embedded_inputs_phoneme, embedded_inputs_tone, embedded_inputs_RPB), 1)
        encoder_outputs = model.encoder.inference(embedded_inputs)
        unit_vector += encoder_outputs.data.cpu().numpy()[0].tolist()

    unit_vector = np.array(unit_vector, dtype=np.float32)
    print('unit_vector shape:', unit_vector.shape)
    unit_vector.tofile(unit_vector_path)

if B:
    unit_vector = []
    unit_vector_path = '../outdir/tacotron2_baseline_medium_modified_v4_0.03_j3/unit_vector/Unit2Vec_UnitVector_test.dat'
    test_file = '../filelists/test_file.lst'
    test_file = np.loadtxt(test_file, 'str')
    files  = filter(lambda i:i.split('.')[0] in test_file, os.listdir(text_dir))

    for fb in tqdm(sorted(files)):
        tqdm.write(fb)
        test_text_path = os.path.join(text_dir, fb)
        sequence = np.loadtxt(test_text_path, 'str')
        sequence = np.array([[phone2id[ph], tone2id[tn], RPB2id[RPB]] for ph, tn, RPB in sequence])[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

        embedded_inputs_phoneme = model.embedding_phoneme(sequence[:,:,0]).transpose(1, 2)
        embedded_inputs_tone = model.embedding_tone(sequence[:,:,1]).transpose(1, 2)
        embedded_inputs_RPB = model.embedding_RPB(sequence[:,:,2]).transpose(1, 2)
        embedded_inputs = torch.cat((embedded_inputs_phoneme, embedded_inputs_tone, embedded_inputs_RPB), 1)
        encoder_outputs = model.encoder.inference(embedded_inputs)
        unit_vector += encoder_outputs.data.cpu().numpy()[0].tolist()

    unit_vector = np.array(unit_vector, dtype=np.float32)
    print('unit_vector shape:', unit_vector.shape)
    unit_vector.tofile(unit_vector_path)

if C:
    assert(hparams.more_information==True)
    # Open more_information = True
    # 另外还要把prenet的dropout变成false
    # 不是 x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
    # 而是 x = F.dropout(F.relu(linear(x)), p=0.5, training=self.training)

    text_dir = '../data/text_addtime'
    train_val_file = '../filelists/train+val_file.lst'
    mel_path = '../data/mel_15ms'
    output_file = '../outdir/tacotron2_baseline_medium_modified_v4_0.03_j3/unit_vector/more_information_dict.pkl'
    train_val_file = np.loadtxt(train_val_file, 'str')
    files  = sorted(list(filter(lambda i:i.split('.')[0] in train_val_file, os.listdir(text_dir))))
    mean_std_file = '../data/MeanStd_Tacotron_mel_15ms.npy'
    mean, std = np.load(mean_std_file)

    more_information_dict = {}
    for fb in tqdm(files):
        fb = fb.split('.')[0]
        tqdm.write(fb)
        test_text_path = os.path.join(text_dir, fb+'.lab')
        sequence = np.loadtxt(test_text_path, 'str')
        sequence = np.array([[phone2id[ph], tone2id[tn], RPB2id[RPB], int(time)] for ph, tn, RPB, time in sequence])[None, :]
        
        text_padded = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
        input_lengths = torch.LongTensor([text_padded.size(1)]).cuda()

        mel_padded = np.load(os.path.join(mel_path, fb+'.npy'))
        mel_padded = np.transpose((mel_padded - mean) / std)
        mel_padded = torch.FloatTensor(mel_padded[None,:,:]).cuda()

        max_len = torch.max(input_lengths.data).item()
        output_lengths = torch.LongTensor([mel_padded.size(2)]).cuda()
        
        x = (text_padded, input_lengths, mel_padded, max_len, output_lengths)
        outputs, more_information_dict[fb] = model(x)

        #mel_outputs_postnet = outputs[1][0].detach().cpu().numpy()
        #plt.matshow(mel_outputs_postnet, origin='lower')
        #plt.colorbar()
        #plt.savefig('test.png')
        #plt.close()

        #audio=recover_wav(mel_outputs_postnet)
        #audio = librosa.util.normalize(audio, norm=np.inf, axis=None)
        #sf.write('test.wav', audio, hparams.sampling_rate, 'PCM_16')
        
    hkl.dump(more_information_dict, output_file, mode='w', compression='gzip')

if extract_info:
    hkl_file = '../outdir//tacotron2_baseline_medium_modified_v4_0.03_j3/unit_vector/more_information_dict.pkl'
    more_information_dict = hkl.load(hkl_file)
    train_val_file = '../filelists/train+val_file.lst'
    train_val_file = np.loadtxt(train_val_file, 'str')

    acoustic_path = '../outdir//tacotron2_baseline_medium_modified_v4_0.03_j3/unit_vector/phone_level_acoustic.dat'
    outputs = []
    for fb in tqdm(sorted(train_val_file)):
        tqdm.write(fb)
        state = more_information_dict[fb]['phone-level acoustic']
        outputs += state.tolist()
    outputs = np.array(outputs, dtype=np.float32)
    print(outputs.shape)
    WriteArrayFloat(acoustic_path, outputs)

    start_mel_path = '../outdir//tacotron2_baseline_medium_modified_v4_0.03_j3/unit_vector/start_mel.dat'
    outputs = []
    for fb in tqdm(sorted(train_val_file)):
        tqdm.write(fb)
        state = more_information_dict[fb]['start mel']
        outputs += state.tolist()
    outputs = np.array(outputs, dtype=np.float32)
    print(outputs.shape)
    WriteArrayFloat(start_mel_path, outputs)

    end_mel_path = '../outdir//tacotron2_baseline_medium_modified_v4_0.03_j3/unit_vector/end_mel.dat'
    outputs = []
    for fb in tqdm(sorted(train_val_file)):
        tqdm.write(fb)
        state = more_information_dict[fb]['end mel']
        outputs += state.tolist()
    outputs = np.array(outputs, dtype=np.float32)
    print(outputs.shape)
    WriteArrayFloat(end_mel_path, outputs)

    corpus_unit_vector_path = '../outdir//tacotron2_baseline_medium_modified_v4_0.03_j3/unit_vector/Unit2Vec_UnitVector.dat'
    corpus_unit_vector_mean_path = '../outdir//tacotron2_baseline_medium_modified_v4_0.03_j3/unit_vector/MinMax_Tacotron2_UnitVector.mean'
    corpus_unit_vector = ReadFloatRawMat(corpus_unit_vector_path, 256)
    minmax_mean = np.c_[np.quantile(corpus_unit_vector, 0.01, axis=0), np.quantile(corpus_unit_vector, 0.99, axis=0)].T
    WriteArrayFloat(corpus_unit_vector_mean_path, minmax_mean)


if C_text:
    assert(hparams.more_information==True)
    # Open more_information = True
    # prenet的dropout依然是True 不然语音没法停下
    # x = F.dropout(F.relu(linear(x)), p=0.5, training=True)

    text_dir = '../data/text'
    test_file = '../filelists/test_file.lst'
    output_file = '../outdir/tacotron2_baseline_medium_modified_v4_0.03_j3/unit_vector/more_information_dict_test.pkl'
    test_file = np.loadtxt(test_file, 'str')
    files  = sorted(list(filter(lambda i:i.split('.')[0] in test_file, os.listdir(text_dir))))

    more_information_dict = {}
    for fb in tqdm(files):
        fb = fb.split('.')[0]
        tqdm.write(fb)
        test_text_path = os.path.join(text_dir, fb+'.lab')
        text = np.loadtxt(test_text_path, 'str')
        sequence = np.array([[phone2id[ph], tone2id[tn], RPB2id[RPB]] for ph, tn, RPB in text])[None, :]
        sequence = torch.autograd.Variable(
            torch.from_numpy(sequence)).cuda().long()

        outputs, more_information_dict[fb] = model.inference(sequence)

        mel_outputs_postnet = outputs[1][0].detach().cpu().numpy()
        #plt.matshow(mel_outputs_postnet, origin='lower')
        #plt.colorbar()
        #plt.savefig('test.png')
        #plt.close()

        audio=recover_wav(mel_outputs_postnet)
        audio = librosa.util.normalize(audio, norm=np.inf, axis=None)
        sf.write('test.wav', audio, hparams.sampling_rate, 'PCM_16')
        
    hkl.dump(more_information_dict, output_file, mode='w', compression='gzip')

if extract_info_test:
    hkl_file = '../outdir/tacotron2_baseline_medium_modified_v4_0.03_j3/unit_vector/more_information_dict_test.pkl'
    more_information_dict = hkl.load(hkl_file)
    test_file = '../filelists/test_file.lst'
    test_file = np.loadtxt(test_file, 'str')

    acoustic_path = '../outdir/tacotron2_baseline_medium_modified_v4_0.03_j3/unit_vector/phone_level_acoustic_test.dat'

    outputs = []
    for fb in tqdm(sorted(test_file)):
        tqdm.write(fb)
        state = more_information_dict[fb]['phone-level acoustic']
        outputs += state.tolist()
    outputs = np.array(outputs, dtype=np.float32)
    print(outputs.shape)
    WriteArrayFloat(acoustic_path, outputs)

if gen_embedding_plot:
    from symbols import phone2id
    import tensorflow as tf
    from tensorflow.examples.tutorials.mnist import input_data
    from tensorflow.contrib.tensorboard.plugins import projector

    LOG_DIR = 'embedding_logs'
    metadata = os.path.join(LOG_DIR, 'metadata.tsv')

    p_points = model.embedding_phoneme.weight.data.cpu().numpy()
    print(p_points.shape)
    UnitVector = tf.Variable(p_points, name='phone')
    id2phone = dict(zip(phone2id.values(), phone2id.keys()))
    with open(metadata, 'w') as metadata_file:
        metadata_file.write('Index\tPhone\n')
        for i in range(p_points.shape[0]):
            metadata_file.write('%d\t%s\n' % (i,id2phone[i]))

    with tf.Session() as sess:
        saver = tf.train.Saver([UnitVector])

        sess.run(UnitVector.initializer)
        saver.save(sess, os.path.join(LOG_DIR, 'phone.ckpt'))

        config = projector.ProjectorConfig()
        # One can add multiple embeddings.
        embedding = config.embeddings.add()
        embedding.tensor_name = UnitVector.name
        # Link this tensor to its metadata file (e.g. labels).
        embedding.metadata_path = 'metadata.tsv'
        # Saves a config file that TensorBoard will read during startup.
        projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)