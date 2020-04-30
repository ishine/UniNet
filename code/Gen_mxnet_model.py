# On Mac OS
import pdb
import numpy as np
import torch,librosa,os

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from train import load_model

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.onnx as torch_onnx
import mxnet as mx
from mxnet import nd
from mxnet.contrib import onnx as onnx_mxnet
from collections import namedtuple

if __name__ == "__main__":
    export_join_model = 1
    export_lstm_cell = 1

    hparams = create_hparams()
    checkpoint_path = '../outdir/tacotron2_baseline_medium_modified_v4_0.03/checkpoint_37440'
    model = load_model(hparams, dev='cpu')
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['state_dict'])
    model.train(False)
    
    if export_join_model:
        join_model = model.decoder.join_model_layer
        x = Variable(torch.FloatTensor([range(256)]))
        y = Variable(torch.FloatTensor([range(128)]))
        print(join_model(x, y))

        model_onnx_path = '../outdir/tacotron2_baseline_medium_modified_v4_0.03/checkpoint_37440_join_model.onnx'
        model_graph = '../outdir/tacotron2_baseline_medium_modified_v4_0.03/checkpoint_37440_join_model_graph'
        output = torch_onnx.export(join_model, 
                                  (x, y), 
                                  model_onnx_path, 
                                  verbose=False)
        print("Export of torch_model.onnx complete!")

        # Import the ONNX model into MXNet's symbolic interface
        sym, arg, aux = onnx_mxnet.import_model(model_onnx_path)
        digraph = mx.viz.plot_network(sym, node_attrs={"shape":"oval","fixedsize":"false"}, save_format = 'png')
        digraph.render(model_graph)
        #print("Loaded torch_model.onnx!")
        #print(sym.get_internals())
        #print(sym.list_arguments(), list(arg))

        data_names = [graph_input for graph_input in sym.list_inputs()
                              if graph_input not in arg and graph_input not in aux]
        print(data_names)
        
        #第一次导入 用ONNX的网络格式
        mod = mx.mod.Module(symbol=sym, data_names=data_names, context=mx.cpu(), label_names=None)
        mod.bind(for_training=False, data_shapes=[('0',(1, 256)),('1', (1, 128))], label_shapes=None)
        mod.set_params(arg_params=arg, aux_params=aux, allow_missing=True, allow_extra=True)
        Batch = namedtuple('Batch', ['data'])
        # forward on the provided data batch
        x = nd.array(x.numpy())
        y = nd.array(y.numpy())
        mod.forward(Batch([x, y]))
        print(mod.get_outputs())

        mxnet_model_path = '../outdir/tacotron2_baseline_medium_modified_v4_0.03/checkpoint_37440_join_model'
        mod.save_checkpoint(mxnet_model_path, 0) 

        # 第二次导入 用MXNet的网络格式
        sym = mx.symbol.load(mxnet_model_path+'-symbol.json') 
        mod=mx.mod.Module(symbol=sym, data_names=data_names, label_names=None)
        mod.bind(data_shapes=[('0',(1, 256)),('1', (1, 128))])
        mod.load_params(mxnet_model_path+'-0000.params')
        mod.forward(Batch([x, y]), is_train=False)
        print(mod.get_outputs())
        print(sym.list_outputs())

        # 第三次导入 用mxnet的C++的引擎
        net = mx.gluon.nn.SymbolBlock.imports(mxnet_model_path+'-symbol.json', ['0', '1'],
                                    param_file=mxnet_model_path+'-0000.params', ctx=mx.cpu()) 
        print(net(x, y))
        
    if export_lstm_cell:
        phone_level_rnn = model.decoder.phone_level_rnn
        x = Variable(torch.FloatTensor([range(256)]))
        h = Variable(torch.FloatTensor([range(128)]))
        c = Variable(torch.FloatTensor([range(128)]))
        print(phone_level_rnn(x, (h, c)))
        
        model_onnx_path = '../outdir/tacotron2_baseline_medium_modified_v4_0.03/checkpoint_37440_phone_level_rnn.onnx'
        model_graph = '../outdir/tacotron2_baseline_medium_modified_v4_0.03/checkpoint_37440_phone_level_rnn_graph'
        output = torch_onnx.export(phone_level_rnn, 
                                  (x, (h, c)), 
                                  model_onnx_path, 
                                  verbose=False)
        print("Export of torch_model.onnx complete!")
        
        # Import the ONNX model into MXNet's symbolic interface
        sym, arg, aux = onnx_mxnet.import_model(model_onnx_path)
        digraph = mx.viz.plot_network(sym, node_attrs={"shape":"oval","fixedsize":"false"}, save_format = 'png')
        digraph.render(model_graph)
        #print("Loaded torch_model.onnx!")
        #print(sym.get_internals())
        #print(sym.list_arguments(), list(arg))

        data_names = [graph_input for graph_input in sym.list_inputs()
                              if graph_input not in arg and graph_input not in aux]
        print(data_names)
        
        #第一次导入 用ONNX的网络格式
        mod = mx.mod.Module(symbol=sym, data_names=data_names, context=mx.cpu(), label_names=None)
        mod.bind(for_training=False, data_shapes=[('input',(1, 256)),('1', (1, 128)),('2', (1, 128))], label_shapes=None)
        mod.set_params(arg_params=arg, aux_params=aux, allow_missing=True, allow_extra=True)
        Batch = namedtuple('Batch', ['data'])
        # forward on the provided data batch
        x = nd.array(x.numpy())
        h = nd.array(h.numpy())
        c = nd.array(c.numpy())
        mod.forward(Batch([x, h, c]))
        print(mod.get_outputs())

        mxnet_model_path = '../outdir/tacotron2_baseline_medium_modified_v4_0.03/checkpoint_37440_phone_level_rnn'
        mod.save_checkpoint(mxnet_model_path, 0) 

        # 第二次导入 用MXNet的网络格式
        sym = mx.symbol.load(mxnet_model_path+'-symbol.json') 
        mod=mx.mod.Module(symbol=sym, data_names=data_names, label_names=None)
        mod.bind(data_shapes=[('input',(1, 256)),('1', (1, 128)),('2', (1, 128))])
        mod.load_params(mxnet_model_path+'-0000.params')
        mod.forward(Batch([x, h, c]), is_train=False)
        print(mod.get_outputs())
        print(sym.list_outputs())

        # 第三次导入 用mxnet的C++的引擎
        net = mx.gluon.nn.SymbolBlock.imports(mxnet_model_path+'-symbol.json', ['input', '1', '2'],
                                    param_file=mxnet_model_path+'-0000.params', ctx=mx.cpu()) 
        print(net(x, h, c))

    if export_dynamic_online_main:
        pass