import numpy as np
import os

phone_set = np.loadtxt('phoneme.lst','str')  # 61种phoneme

phone2id = {ph:i for i,ph in enumerate(phone_set)}

tone2id  = {'0'   : 0, # 结构轻声
            '1'   : 1, 
            '2'   : 2, 
            '3'   : 3, 
            '4'   : 4, 
            '6'   : 5, # 语调轻声
            '7'   : 5, # 语调轻声
            '8'   : 5, # 语调轻声
            '9'   : 5, # 语调轻声
            'XX'  : 6, # 短停顿   sp
            'sil' : 7} # 首尾静音 sil

RPB2id   =  {str(i):i for i in range(10)}
RPB2id['sil'] = 10