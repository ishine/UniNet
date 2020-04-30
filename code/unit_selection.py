# -*- coding: utf-8 -*-
import sys
sys.path.append('../tools')
import numpy as np
import torch,os
from torch.nn import functional as F
from copy import deepcopy
import hickle as hkl
import pdb
from hparams import create_hparams
from model import Tacotron2
from train import load_model
from symbols import phone2id, tone2id, RPB2id

from tools import *
from functions import *
import pdb

class PhoneUnit:
    def __init__(self, my_dict):
        self.phoneID = my_dict['phoneID']
        self.UnitNo = my_dict['UnitNo']
        self.UnitVectorNo = my_dict['UnitVectorNo']
        self.SenNo = my_dict['SenNo']
        self.PhoneNo = my_dict['PhoneNo']
        self.SegBegFrm = my_dict['SegBegFrm']
        self.SegEndFrm = my_dict['SegEndFrm']

def read_vtb_file(path):
    lis = []
    with open(path,'rt') as fp:
        fp.readline()
        for idx,line in enumerate(fp):
            line = line.split()
            phone = line[0]
            cand_num = int(line[1])
            cand_units_info = map(lambda i:i[1:-1].split(','), line[2:])
            units_lis = []
            for u in cand_units_info:
                unit_dict = {}
                unit_dict['phoneID'] = int(u[0])
                unit_dict['UnitNo'] = int(u[1])
                unit_dict['UnitVectorNo'] = int(u[2])
                unit_dict['SenNo'] = u[3]
                unit_dict['PhoneNo'] = int(u[4])
                unit_dict['SegBegFrm'] = int(u[5])
                unit_dict['SegEndFrm'] = int(u[6])
                units_lis.append(PhoneUnit(unit_dict))
            lis.append({'phone':phone, 'cand_num':cand_num, 'units_info':units_lis})
    return lis

def read_lab_file(path):
    lis = []
    sequence = np.loadtxt(path, 'str')
    sequence = np.array([[phone2id[ph], tone2id[tn], RPB2id[RPB]] for ph, tn, RPB in sequence])[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    embedded_inputs_phoneme = model.embedding_phoneme(sequence[:,:,0]).transpose(1, 2)
    embedded_inputs_tone = model.embedding_tone(sequence[:,:,1]).transpose(1, 2)
    embedded_inputs_RPB = model.embedding_RPB(sequence[:,:,2]).transpose(1, 2)
    embedded_inputs = torch.cat((embedded_inputs_phoneme, embedded_inputs_tone, embedded_inputs_RPB), 1)
    encoder_outputs = model.encoder.inference(embedded_inputs)
    unit_vector = encoder_outputs.data.cpu().numpy()[0]
    sequence = sequence[0]
    for i in range(len(sequence)):
        my_dict = {}
        my_dict['lab'] = sequence[i]
        my_dict['unit_vector'] = unit_vector[i]
        lis.append(my_dict)
    return lis

def cal_join_model(cur_linguitsic, lstm_input, lstm_states):
    #pdb.set_trace()
    cur_linguitsic = torch.autograd.Variable(torch.from_numpy(cur_linguitsic)).cuda().float().reshape(1,-1)
    lstm_input = torch.autograd.Variable(torch.from_numpy(lstm_input)).cuda().float().reshape(1,-1)
    lstm_states = torch.autograd.Variable(torch.from_numpy(lstm_states)).cuda().float().reshape(2, 1,-1)
    h, c = model.decoder.phone_level_rnn(lstm_input, (lstm_states[0], lstm_states[1]))
    lstm_states = np.asarray((h.data.cpu().numpy().flatten(), c.data.cpu().numpy().flatten()))
    JoinModel_output = model.decoder.join_model_layer(cur_linguitsic, h)
    JoinModel_output = JoinModel_output.data.cpu().numpy().flatten()
    #pdb.set_trace()
    return JoinModel_output, lstm_states

def euclidean_dist(u,v):
    return np.sqrt(np.sum(np.square(u-v)))

def change_lattice_structure(lattice, cur_position_x, best_k_pos_lis):
    lattice_bak = deepcopy(lattice)
    for j in range(lattice[cur_position_x]['cand_num']):
        for k in range(cur_position_x):
            lattice_bak[k]['units_info'][j] = lattice[k]['units_info'][best_k_pos_lis[j]]
    return lattice_bak

def output_frm_file(frm_path, units_info, g_nCutSil=True):
    # g_nCutSil代表是否对句子头尾的静音做截断处理  
    with open(frm_path, 'wt') as f:
        for idx,unit in enumerate(units_info):
            SenNo = unit.SenNo
            s = unit.SegBegFrm
            e = unit.SegEndFrm
            if g_nCutSil:
                if idx == 0:
                    if e-s>9:
                        s = e-10
                if idx == len(units_info)-1:
                    if e-s>9:
                        e = s+10

            for t in range(s,e):
                f.write('%s %s %0.2f\n' % (SenNo, t, 1.00))

def unit_selection(target_units, lattice, target_weight, join_weight, export_dir, filebasename):
    pPathEva = np.zeros(max_cand_num)
    lstm_state = np.zeros((max_cand_num, 2, phone_level_lstm_states_dim))
    for i in range(len(target_units)):
        cur_cand_num = lattice[i]['cand_num']
        # 计算目标代价
        if i == 0:
            target_costs = np.zeros(cur_cand_num)
            for j in range(cur_cand_num):
                u = lattice[i]['units_info'][j]
                target_costs[j] = euclidean_dist(target_units[i]['unit_vector'], corpus_embedding[u.UnitVectorNo])
                pPathEva[j] = target_costs[j]
        else:
            pre_cand_num = lattice[i-1]['cand_num']
            pre_join_mat = np.zeros((pre_cand_num, embedded_dims))
            for k in range(pre_cand_num):
                u_p = lattice[i-1]['units_info'][k]
                pre_join_mat[k], lstm_state[k] = cal_join_model(target_units[i]['unit_vector'],
                                                 corpus_more_information[u_p.SenNo]['phone-level acoustic'][u_p.PhoneNo],
                                                 lstm_state[k])

            target_costs = np.zeros(cur_cand_num)
            join_costs = np.zeros((cur_cand_num, pre_cand_num))
            best_k_pos_lis= np.zeros(cur_cand_num, dtype=np.int)
            for j in range(cur_cand_num):
                u = lattice[i]['units_info'][j]
                target_costs[j] = euclidean_dist(target_units[i]['unit_vector'], corpus_embedding[u.UnitVectorNo])

                fBestPathEva = float('inf')
                for k in range(pre_cand_num):
                    u_p = lattice[i-1]['units_info'][k]
                    join_costs[j][k] = euclidean_dist(pre_join_mat[k],
                                                      corpus_more_information[u.SenNo]['phone-level join model output'][u.PhoneNo])

                    join_costs[j][k] += euclidean_dist(corpus_more_information[u_p.SenNo]['frame-level end prenet'][u_p.PhoneNo],
                                                       corpus_more_information[u.SenNo]['frame-level start prenet'][u.PhoneNo])
                    
                    fPathEva = pPathEva[k] + target_weight * target_costs[j] + join_weight *join_costs[j][k]
                    if fPathEva < fBestPathEva:
                        nBestPathNo = k
                        fBestPathEva = fPathEva
                pPathEva[j] = fBestPathEva
                best_k_pos_lis[j] = nBestPathNo

            lattice = change_lattice_structure(lattice, i, best_k_pos_lis)

    best_path_idx = np.argmin(pPathEva)
    best_units_info = [lattice[i]['units_info'][best_path_idx] for i in range(nTrgPhoneNum)]

    audio_path = os.path.join(export_dir, filebasename + '.wav')
    frm_path = os.path.join(export_dir, filebasename + '.frm')
    output_frm_file(frm_path, best_units_info, g_nCutSil=True)
    os.system('wine Concatenate.exe -a ../data/audio -f %s -o %s' % (frm_path, audio_path))

if __name__ == "__main__":
    print('Loading Model ...')
    hparams = create_hparams()
    checkpoint_path = "../outdir/tacotron2_baseline_medium_modified_v3/checkpoint_28080"
    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.cuda().eval()

    print('Loading Corpus Units ...')
    corpus_embedding_path = '../outdir/tacotron2_baseline_medium_modified_v3/unit_vector/Unit2Vec_UnitVector.dat'
    embedded_dims = 256
    phone_level_lstm_states_dim=128
    corpus_embedding = ReadFloatRawMat(corpus_embedding_path, embedded_dims)
    corpus_more_information_path = '../outdir/tacotron2_baseline_medium_modified_v3/unit_vector/more_information_dict.pkl'
    corpus_more_information = hkl.load(corpus_more_information_path)

    test_list = np.loadtxt('../filelists/test_file.lst', 'str')
    #vtb_dir = './Discover_embedding/tacotron_gened_vtb_restrict_tone_normdis'
    vtb_dir = '../data/vtb'
    text_dir = '../data/text'
    max_cand_num = 25

    export_dir = '../outdir/tacotron2_baseline_medium_modified_v3/GenAudio_all_us_hmmpresel_1_1'

    SaveMkdir(export_dir)
    for fb in test_list:
        print('Synthesis %s ...' % fb)
        target_units = read_lab_file(os.path.join(text_dir, fb+'.lab'))
        lattice = read_vtb_file(os.path.join(vtb_dir, fb+'.vtb'))
        nTrgPhoneNum = len(target_units)
        assert(max([lattice[i]['cand_num'] for i in range(len(lattice))])==max_cand_num)
        unit_selection(target_units, lattice, 1, 1, export_dir, fb)
