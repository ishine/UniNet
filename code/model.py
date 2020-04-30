# -*- coding: utf-8 -*-
from math import sqrt
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from layers import ConvNorm, LinearNorm
from utils import to_gpu, get_mask_from_lengths, get_mask_from_times, get_start_end_times
import numpy as np


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        # attention_kernel_size是31
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        # attention_weights_cat.shape (B,2,72)
        # processed_attention (3,32,72)
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        #processed_attention (3,72,128)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, query_dim, keys_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        # 传统attention需要query和keys做线性变换再v^T.*tanh(W * query + V * keys)
        # 这个query_layer和memory_layer分别得到 W * query 和 V * keys
        # w_init_gain='tanh'是因为他们包在tanh(W * query + V * keys)函数中
        self.query_layer = LinearNorm(query_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(keys_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        # 当前attention除了传统参数还包括对注意力权重做卷积处理
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.EOS_embedding_layer = nn.Embedding(1, attention_dim)

        self.v = LinearNorm(attention_dim, 1, bias=False)

        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, query_dim)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        # processed_query经广播机制与processed_attention_weights和processed_memory相加
        # processed_attention_weights这个注意力权重是和加性attention不一样的地方
        # energies (3,72,1)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))
        # energies (3,72)
        energies = energies.squeeze(-1)
        return energies

    def forward(self, query, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        query: attention rnn last output
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded text data
        """

        #get_alignment_energies即得到注意力权重alpha
        alignment = self.get_alignment_energies(
            query, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)

        return attention_weights


class SimpleAttention(nn.Module):
    def __init__(self, query_dim, keys_dim, attention_dim):
        super(SimpleAttention, self).__init__()
        # 传统attention需要query和keys做线性变换再v^T.*tanh(W * query + V * keys)
        # 这个query_layer和memory_layer分别得到 W * query 和 V * keys
        # w_init_gain='tanh'是因为他们包在tanh(W * query + V * keys)函数中
        self.query_layer = LinearNorm(query_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(keys_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory):
        """
        PARAMS
        ------
        query: decoder output (batch, query_dim)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)

        RETURNS
        -------
        alignment (batch, max_time)
        """
        processed_query = self.query_layer(query.unsqueeze(1))
        # processed_query经广播机制与processed_attention_weights和processed_memory相加
        # processed_attention_weights这个注意力权重是和加性attention不一样的地方
        # energies (3,72,1)
        energies = self.v(torch.tanh(
            processed_query + processed_memory))
        # energies (3,72)
        energies = energies.squeeze(-1)
        return energies

    def forward(self, query, processed_memory, mask):
        """
        PARAMS
        ------
        query: attention rnn last output
        processed_memory: processed encoder outputs
        mask: binary mask for padded text data
        """

        #get_alignment_energies即得到注意力权重alpha
        alignment = self.get_alignment_energies(
            query, processed_memory)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        #attention_weights = F.log_softmax(alignment, dim=1)

        return alignment


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
        	#单元挑选等信息生成时需要关掉dropout(True -> self.training)
        	#SPSS时打开（保持一个随机性，语音stop token不会预测错误，静音更干净）
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class JoinModol(nn.Module):
    def __init__(self, linguistic_dim, phone_level_rnn_dim, join_model_dim, join_model_hidden_dim):
        super(JoinModol, self).__init__()
        self.layers1 = LinearNorm(linguistic_dim + phone_level_rnn_dim, join_model_hidden_dim)
        self.layers2 = LinearNorm(join_model_hidden_dim, join_model_hidden_dim)
        self.layers3 = LinearNorm(join_model_hidden_dim, join_model_dim)

    def forward(self, linguistic, state):
        x = torch.cat((linguistic, state), -1)
        x = F.relu(self.layers1(x))
        x = F.relu(self.layers2(x))
        x = self.layers3(x)
        return x


class vectorBased_selfAttention(nn.Module):
    # inputs (T, N)
    # Output (N)
    # input_dims == N
    def __init__(self, input_dims, hidden_dims):
        super(vectorBased_selfAttention, self).__init__()
        self.layers1 = LinearNorm(input_dims, hidden_dims)
        self.layers2 = LinearNorm(hidden_dims, input_dims)

    def forward(self, inputs):
        h = F.relu(self.layers1(inputs))
        p = F.softmax(self.layers2(h), dim = 0)
        x = torch.sum(torch.mul(inputs, p), dim = 0)
        return x


class annealing(nn.Module):
    def __init__(self, ratio):
        super(annealing, self).__init__()
        self.ratio = ratio
        self.x = 1
        self.y = 1

    def forward(self):
        self.x = self.x * self.ratio
        self.y = 0.5 * self.x + 0.5

    def set_annealing_x(self, x):
        self.x = x

    def get_annealing_x(self):
        return self.x

    def get_annealing_y(self):
        return self.y


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_mel_channels, hparams.postnet_embedding_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.postnet_embedding_dim))
        )

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_embedding_dim,
                             hparams.postnet_embedding_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int((hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_embedding_dim, hparams.n_mel_channels,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams.n_mel_channels))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, hparams):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hparams.encoder_embedding_dim,
                         hparams.encoder_embedding_dim,
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hparams.encoder_embedding_dim,
                            int(hparams.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = F.relu(conv(x))

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.frame_level_rnn_dim = hparams.frame_level_rnn_dim
        self.phone_level_rnn_dim = hparams.phone_level_rnn_dim
        self.join_model_dim = hparams.frame_level_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.p_decoder_dropout = hparams.p_decoder_dropout
        self.decoder_training_mode = hparams.decoder_training_mode
        if self.decoder_training_mode == 'random annealing':
            self.annealing = annealing(0.9999)

        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,
            [hparams.prenet_dim, hparams.prenet_dim])

        self.frame_level_rnn = nn.LSTMCell(hparams.prenet_dim, hparams.frame_level_rnn_dim)

        self.self_attention = vectorBased_selfAttention(hparams.frame_level_rnn_dim, hparams.self_attention_dim)

        self.text_attention_layer = SimpleAttention(
            hparams.frame_level_rnn_dim,
            hparams.encoder_embedding_dim,
            hparams.attention_dim)

        self.phone_level_rnn = nn.LSTMCell(hparams.frame_level_rnn_dim,
                                           hparams.phone_level_rnn_dim)

        self.join_model_layer = JoinModol(hparams.encoder_embedding_dim, hparams.phone_level_rnn_dim,
                                          self.join_model_dim, hparams.join_model_hidden_dim)

        self.decoder_rnn = nn.LSTMCell(
            hparams.frame_level_rnn_dim + self.join_model_dim,
            hparams.decoder_rnn_dim)

        self.linear_projection = LinearNorm(
            hparams.decoder_rnn_dim + self.join_model_dim,
            hparams.n_mel_channels * hparams.n_frames_per_step)

        self.attention_layer = Attention(
            hparams.decoder_rnn_dim + self.join_model_dim,
            hparams.encoder_embedding_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size)

        self.max_decoder_steps = hparams.max_decoder_steps

        if hparams.more_information:
            self.more_information = True
        else:
            self.more_information = False

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, text_mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, phone-level join model output, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        """
        B = memory.size(0)
        self.MAX_TIME = memory.size(1)

        self.frame_level_rnn_hidden = Variable(memory.data.new(
            B, self.frame_level_rnn_dim).zero_())
        self.frame_level_rnn_cell = Variable(memory.data.new(
            B, self.frame_level_rnn_dim).zero_())
        self.frame_level_rnn_hidden_list = [[] for _ in range(B)]

        self.phone_level_rnn_hidden = Variable(memory.data.new(
            B, self.phone_level_rnn_dim).zero_())
        self.phone_level_rnn_cell = Variable(memory.data.new(
            B, self.phone_level_rnn_dim).zero_())

        self.overall_frame_level_rnn_hidden = Variable(memory.data.new(
            B, self.frame_level_rnn_dim).zero_())

        self.join_model_output = Variable(memory.data.new(
            B, self.frame_level_rnn_dim).zero_())

        # 计算单元挑选损失所必需的的一对变量
        self.join_outs = Variable(memory.data.new(
            B, self.MAX_TIME, self.frame_level_rnn_dim).zero_())
        self.acoustics_of_phone = Variable(memory.data.new(
            B, self.MAX_TIME, self.frame_level_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        self.current_memory = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, self.MAX_TIME + 1).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, self.MAX_TIME + 1).zero_())

        # text_attention_layer include EOS
        eos_embedding = self.attention_layer.EOS_embedding_layer(
                                    Variable(memory.data.new(B).zero_()).long())\
                                    .unsqueeze(2).transpose(1, 2)
        processed_memory = self.attention_layer.memory_layer(memory)
        self.processed_memory = torch.cat((processed_memory, eos_embedding), 1)

        # Not include EOS
        self.text_processed_memory = self.text_attention_layer.memory_layer(memory)
        self.text_alignment = Variable(memory.data.new(
            B, self.MAX_TIME, self.MAX_TIME).zero_()) - float('inf')

        # self.mask是随着音素的位置动态改变的 用了get_mask_from_times
        self.mask = Variable(memory.data.new(B, self.MAX_TIME + 1).zero_()).type(dtype=torch.uint8)
        # self.text_mask可能是None也能不是    用了get_mask_from_lengths
        self.text_mask = text_mask

        if self.more_information:
            self.more_information_dict = {}
            self.more_information_dict['start mel'] = []
            self.more_information_dict['end mel'] = []
            self.more_information_dict['phone-level acoustic'] = []
            #self.more_information_dict['decoder lstm hidden'] = []

    def initialize_phone_start_states(self, memory, phone_start_idx, cur_frame_index):
        B = memory.size(0)
        #如该帧是音素开头，则start_pos_mask=0
        start_pos_mask = Variable(memory.data.new(B, 1).zero_()) + 1
        for j in range(B):
            phone_idx = np.where(cur_frame_index == phone_start_idx[j])[0]
            # phone_idx 0（不是首帧，忽略） 或 1 或 很多（无效帧，忽略）
            if len(phone_idx) == 1:
                phone_idx = phone_idx[0]
                start_pos_mask[j] = 0
                self.current_memory[j,:] = memory[j,phone_idx,:]
                self.mask[j,:] = get_mask_from_times(phone_idx, self.MAX_TIME + 1)
                self.frame_level_rnn_hidden_list[j] = []

        #start_pos是首帧的位置
        start_pos = np.where(start_pos_mask.cpu().numpy() == 0)[0]
        if len(start_pos) > 0:
            self.phone_level_rnn_hidden[start_pos,:], self.phone_level_rnn_cell[start_pos,:] = \
                                                        self.phone_level_rnn(
                                                            self.overall_frame_level_rnn_hidden[start_pos,:],
                                                           (self.phone_level_rnn_hidden[start_pos,:],
                                                            self.phone_level_rnn_cell[start_pos,:]))
            self.join_model_output[start_pos,:] = self.join_model_layer(self.current_memory[start_pos,:],
                                                                        self.phone_level_rnn_hidden[start_pos,:])

            self.frame_level_rnn_hidden = self.frame_level_rnn_hidden * start_pos_mask
            self.frame_level_rnn_cell = self.frame_level_rnn_cell * start_pos_mask

    def initialize_phone_end_states(self, B, phone_end_idx, cur_frame_index, mode):
        #如该帧是音素结尾，则end_pos_mask=0
        end_pos_mask = Variable(self.decoder_hidden.data.new(B, 1).zero_()) + 1
        phone_idx_lst = []
        for j in range(B):
            union_phone_end_idx = np.unique(phone_end_idx[j])
            phone_idx = np.where(cur_frame_index == union_phone_end_idx)[0]
            # phone_idx 0（不是为尾帧，忽略） 或 1
            if len(phone_idx) == 1:
                phone_idx_lst.append(phone_idx[0])
                all_hiddens_in_phone = torch.stack(self.frame_level_rnn_hidden_list[j]) # (T, N)
                all_hiddens_in_phone = self.self_attention(all_hiddens_in_phone)        # (N,  )
                self.overall_frame_level_rnn_hidden[j, :] = all_hiddens_in_phone
                if mode == 'forward':
                    self.acoustics_of_phone[j, phone_idx[0], :] = all_hiddens_in_phone
                    self.join_outs[j, phone_idx[0], :] = self.join_model_output[j,:]
                end_pos_mask[j] = 0

        #end_pos是尾帧的位置
        if mode == 'forward':
            end_pos = np.where(end_pos_mask.cpu().numpy() == 0)[0]
            if len(end_pos) > 0:
                mask = None if self.text_mask is None else self.text_mask[end_pos]
                alpha = self.text_attention_layer(
                    self.overall_frame_level_rnn_hidden[end_pos,:],
                    self.text_processed_memory[end_pos],
                    mask)
                for i in range(len(end_pos)):
                    self.text_alignment[end_pos[i], phone_idx_lst[i], :] = alpha[i]

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (T_out, B, n_mel_channels)

        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:

        RETURNS
        -------
        mel_outputs:
        """
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        # view只能用在contiguous的variable上
        # 如果在view之前用了transpose, permute等，需要用contiguous()来返回一个连续分配的内存形式
        # mel_outputs[0] (3, 80) torch.stack(mel_outputs) (1000, 3, 80)
        alignments = torch.stack(alignments).transpose(0, 1)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)
        # mel_outputs (3,80,1000)
        return mel_outputs, alignments

    def decode(self, decoder_input):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        """
        # decoder_input (3, 128)
        # cell_input.shape (3,256+128)
        self.frame_level_rnn_hidden, self.frame_level_rnn_cell = self.frame_level_rnn(
            decoder_input, (self.frame_level_rnn_hidden, self.frame_level_rnn_cell))

        B = self.frame_level_rnn_hidden.size(0)
        for i in range(B):
            self.frame_level_rnn_hidden_list[i] += [self.frame_level_rnn_hidden[i]]

        # decoder_input (3,768)
        decoder_input = torch.cat((self.frame_level_rnn_hidden, self.join_model_output), -1)

        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_join_model_output = torch.cat(
            (self.decoder_hidden, self.join_model_output), dim=1)
        
        decoder_output = self.linear_projection(
            decoder_hidden_join_model_output)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)

        self.attention_weights = self.attention_layer(
            decoder_hidden_join_model_output,
            self.processed_memory,
            attention_weights_cat, self.mask)

        self.attention_weights_cum += self.attention_weights

        return decoder_output, self.attention_weights

    def forward(self, phone_start_idx, phone_end_idx, memory, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        phone_start_idx: 音素边界的首帧索引
        phone_end_idx: 音素边界的尾帧索引
        memory: Encoder outputs
        decoder_inputs: 文本对应的完整梅尔谱
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        """
        B = memory.size(0)

        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        # reshape decoder_inputs的维度
        # decoder_input  (1,3,80)
        self.initialize_decoder_states(memory, text_mask = ~get_mask_from_lengths(memory_lengths))

        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        # decoder_inputs (1000,3,80)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        # decoder_inputs (1001,3,80)

        #保证自然单元的帧连接代价是0
        if self.more_information:
            assert(B == 1)
            for i in range(decoder_inputs.size(0)):
                if i in phone_start_idx[0]:
                    self.more_information_dict['start mel'].append(decoder_inputs[i][0].data.cpu())
                if i-1 in phone_end_idx[0]:
                    self.more_information_dict['end mel'].append(decoder_inputs[i][0].data.cpu())

        decoder_inputs = self.prenet(decoder_inputs)
        # decoder_inputs (1001,3,128)

        mel_outputs, alignments = [], []
        for i in range(decoder_inputs.size(0) - 1):
            self.initialize_phone_start_states(memory, phone_start_idx, i)
            
            if self.decoder_training_mode == 'teacher forcing':
                decoder_input = decoder_inputs[len(mel_outputs)]
            elif self.decoder_training_mode == 'random annealing':
                if i == 0:
                    decoder_input = decoder_inputs[0]
                else:
                    annealing_y = self.annealing.get_annealing_y()
                    index = np.random.choice(2, 1, p=[1-annealing_y, annealing_y])[0]
                    if index == 1:
                        decoder_input = decoder_inputs[len(mel_outputs)] #强制教学
                    else:
                        decoder_input = self.prenet(mel_output)
            else:
                print('No such decoder_training_mode. Check hparams.py!')
                exit()

            mel_output, attention_weights = self.decode(decoder_input)
            # mel_output (3, 80)
            # mel_output.squeeze(1) (3, 80)
            mel_outputs += [mel_output.squeeze(1)]
            alignments += [attention_weights]

            self.initialize_phone_end_states(B, phone_end_idx, i, mode = 'forward')

            if self.more_information:
                assert(B == 1)
                if i in phone_end_idx[0]:
                    self.more_information_dict['phone-level acoustic'].append(self.overall_frame_level_rnn_hidden[0].data.cpu())

        mel_outputs, alignments = self.parse_decoder_outputs(mel_outputs, alignments)

        if self.more_information:
            return mel_outputs, alignments, self.acoustics_of_phone, self.join_outs, self.text_alignment, self.more_information_dict
        else:
            return mel_outputs, alignments, self.acoustics_of_phone, self.join_outs, self.text_alignment

    def inference(self, phone_start_idx, phone_end_idx, memory):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        """
        B = memory.size(0)

        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, text_mask = None)

        mel_outputs, alignments = [], []
        if len(phone_end_idx[0]) > 0:
            #输入包括了时间
            Input_Given_timeQ = True
            self.max_decoder_steps = phone_end_idx[0][-1] + 1
        else:
            Input_Given_timeQ = False
            phone_num = 0

        #Force alignment
        #input_mel = np.load('ref_mel.npy')
        #input_mel = torch.FloatTensor(input_mel[None,:,:]).cuda()
        #for i in range(input_mel.shape[2]):
        for i in range(self.max_decoder_steps):
            self.initialize_phone_start_states(memory, phone_start_idx, i)

            decoder_input = self.prenet(decoder_input)
            mel_output, attention_weights = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            alignments += [attention_weights]

            decoder_input = mel_output
            #Force alignment
            #decoder_input = input_mel[:,:,i]

            if not Input_Given_timeQ:
                mask_pos = np.where(attention_weights[0].data.cpu().numpy() > 0)[0]
                if attention_weights[0, mask_pos[1]] > 0.5:
                    #跳转音素
                    phone_end_idx = np.c_[phone_end_idx, [[i]]]
                    phone_start_idx = np.c_[phone_start_idx, [[i+1]]]
                    phone_num += 1
                if i == self.max_decoder_steps - 1:
                    print("Warning! Reached max decoder steps")

            self.initialize_phone_end_states(B, phone_end_idx, i, mode='inference')
            if self.more_information:
                assert(B == 1)
                if i in phone_end_idx[0]:
                    self.more_information_dict['phone-level acoustic'].append(self.overall_frame_level_rnn_hidden[0].data.cpu())

            if phone_num == memory.size(1):
                break

        mel_outputs, alignments = self.parse_decoder_outputs(mel_outputs, alignments)

        if self.more_information:
            return mel_outputs, alignments, self.more_information_dict
        else:
            return mel_outputs, alignments


class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.frame_level_rnn_dim = hparams.frame_level_rnn_dim
        self.n_frames_per_step = hparams.n_frames_per_step

        self.embedding_phoneme = nn.Embedding(
            hparams.n_symbols_phoneme, hparams.symbols_embedding_dim_phoneme)
        std = sqrt(2.0 / (hparams.n_symbols_phoneme + hparams.symbols_embedding_dim_phoneme))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding_phoneme.weight.data.uniform_(-val, val)

        self.embedding_tone = nn.Embedding(
            hparams.n_symbols_tone, hparams.symbols_embedding_dim_tone)
        std = sqrt(2.0 / (hparams.n_symbols_tone + hparams.symbols_embedding_dim_tone))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding_tone.weight.data.uniform_(-val, val)

        self.embedding_RPB = nn.Embedding(
            hparams.n_symbols_RPB, hparams.symbols_embedding_dim_RPB)
        std = sqrt(2.0 / (hparams.n_symbols_RPB + hparams.symbols_embedding_dim_RPB))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding_RPB.weight.data.uniform_(-val, val)
        
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)
        self.more_information = hparams.more_information

    def parse_batch(self, batch):
        text_padded, text_alignment_padded, input_lengths, mel_padded, alignments_padded, alignments_weights_padded, \
            output_lengths = batch
        text_padded = to_gpu(text_padded).long()
        text_alignment_padded = to_gpu(text_alignment_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        alignments_padded = to_gpu(alignments_padded).float()
        alignments_weights_padded = to_gpu(alignments_weights_padded).float()
        output_lengths = to_gpu(output_lengths).long()

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths),
            (mel_padded, alignments_padded, alignments_weights_padded, text_alignment_padded))

    def parse_output(self, outputs, text_lengths=None, output_lengths=None):
        if self.mask_padding and text_lengths is not None and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)

            mask1 = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask1 = mask1.permute(1, 0, 2)
            outputs[0].data.masked_fill_(mask1, 0.0)
            outputs[1].data.masked_fill_(mask1, 0.0)

            mask2 = mask.expand(max(text_lengths).item() + 1, mask.size(0), mask.size(1))
            mask2 = mask2.permute(1, 2, 0)
            outputs[2].data.masked_fill_(mask2, 0.0)

            mask = ~get_mask_from_lengths(text_lengths)
            mask3 = mask.expand(self.frame_level_rnn_dim, mask.size(0), mask.size(1))
            mask3 = mask3.permute(1, 2, 0)
            outputs[3].data.masked_fill_(mask3, 0.0)
            outputs[4].data.masked_fill_(mask3, 0.0)

        return outputs

    def forward(self, inputs):
        # mels是文本对应的完整梅尔谱
        text_inputs, text_lengths, mels, max_len, output_lengths = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        embedded_inputs_phoneme = self.embedding_phoneme(text_inputs[:,:,0]).transpose(1, 2)
        embedded_inputs_tone = self.embedding_tone(text_inputs[:,:,1]).transpose(1, 2)
        embedded_inputs_RPB = self.embedding_RPB(text_inputs[:,:,2]).transpose(1, 2)
        embedded_inputs = torch.cat((embedded_inputs_phoneme, embedded_inputs_tone, embedded_inputs_RPB), 1)

        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        phone_start_idx, phone_end_idx = get_start_end_times(text_inputs[:,:,3].cpu().numpy())
        '''
        text_inputs[:,:,3].cpu().numpy()
        [[38  8  6  6 17 19 12  6 15  4 12 11  5  8  4  8  7 14 24  6  7  4  9  6
           8  6  7  8  5  8 14  3  4 11  3  9  3 12 20 19]
         [41  8 10  6  6  7 11 11  6 14  5  4 10  8  6 13  6 10  5  8  7 10  5 12
          22  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
         [43  6 10  8 13 17 10 11  3 12  9  6  5 12 12  4 16 15  0  0  0  0  0  0
           0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]]

        phone_start_idx
        [[  0  38  46  52  58  75  94 106 112 127 131 143 154 159 167 171 179 186
          200 224 230 237 241 250 256 264 270 277 285 290 298 312 315 319 330 333
          342 345 357 377]
         [  0  41  49  59  65  71  78  89 100 106 120 125 129 139 147 153 166 172
          182 187 195 202 212 217 229 251(无效) 251(无效) 251(无效) 251(无效) 251(无效...) 251 251 251 251 251 251
          251 251 251 251]
         [  0  43  49  59  67  80  97 107 118 121 133 142 148 153 165 177 181 197
          212(无效) 212(无效) 212(无效...) 212 212 212 212 212 212 212 212 212 212 212 212 212 212 212
          212 212 212 212]]

        phone_end_idx
        [[ 37  45  51  57  74  93 105 111 126 130 142 153 158 166 170 178 185 199
          223 229 236 240 249 255 263 269 276 284 289 297 311 314 318 329 332 341
          344 356 376 395]
         [ 40  48  58  64  70  77  88  99 105 119 124 128 138 146 152 165 171 181
          186 194 201 211 216 228 250(有效) 250(有效) 250(有效) 250(有效...) 250 250 250 250 250 250 250 250
          250 250 250 250]
         [ 42  48  58  66  79  96 106 117 120 132 141 147 152 164 176 180 196 211(有效)
          211(有效) 211(有效...) 211 211 211 211 211 211 211 211 211 211 211 211 211 211 211 211
          211 211 211 211]]
        '''

        if self.more_information:
            mel_outputs, alignments, acoustics_of_phone, join_outs, text_alignment, more_information_dict = self.decoder(
                                                                phone_start_idx,
                                                                phone_end_idx, encoder_outputs, mels,
                                                                memory_lengths=text_lengths)
        else:
            mel_outputs, alignments, acoustics_of_phone, join_outs, text_alignment = self.decoder(phone_start_idx,
                                                                phone_end_idx, encoder_outputs, mels,
                                                                memory_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        if self.more_information:
            more_information_dict['start mel'] = torch.stack(more_information_dict['start mel']).numpy()
            more_information_dict['end mel'] = torch.stack(more_information_dict['end mel']).numpy()
            more_information_dict['phone-level acoustic'] = torch.stack(more_information_dict['phone-level acoustic']).numpy()

            return [self.parse_output(
                [mel_outputs, mel_outputs_postnet, alignments, acoustics_of_phone, join_outs, text_alignment],
                text_lengths, output_lengths), more_information_dict]
        else:
            return self.parse_output(
                [mel_outputs, mel_outputs_postnet, alignments, acoustics_of_phone, join_outs, text_alignment],
                text_lengths, output_lengths)

    def inference(self, inputs):
        embedded_inputs_phoneme = self.embedding_phoneme(inputs[:,:,0]).transpose(1, 2)
        embedded_inputs_tone = self.embedding_tone(inputs[:,:,1]).transpose(1, 2)
        embedded_inputs_RPB = self.embedding_RPB(inputs[:,:,2]).transpose(1, 2)
        embedded_inputs = torch.cat((embedded_inputs_phoneme, embedded_inputs_tone, embedded_inputs_RPB), 1)

        encoder_outputs = self.encoder.inference(embedded_inputs)

        if inputs.size(2) == 4:
            #文本输入包含时间信息
            phone_start_idx, phone_end_idx = get_start_end_times(inputs[:,:,3].cpu().numpy())
        else:
            #文本输入不包含时间信息
            phone_start_idx = np.array([[0]])
            phone_end_idx = np.array([[]])
        
        if self.more_information:
            mel_outputs, alignments, more_information_dict = self.decoder.inference(
                                                                    phone_start_idx, phone_end_idx, encoder_outputs)
        else:
            mel_outputs, alignments = self.decoder.inference(phone_start_idx, phone_end_idx, encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        if self.more_information:
            more_information_dict['phone-level acoustic'] = torch.stack(more_information_dict['phone-level acoustic']).numpy()

            return [self.parse_output(
                [mel_outputs, mel_outputs_postnet, alignments]), more_information_dict]
        else:
            return self.parse_output(
                [mel_outputs, mel_outputs_postnet, alignments])


if __name__ == "__main__":
    from hparams import create_hparams
    hparams = create_hparams()
    model = Tacotron2(hparams)

    text_padded = torch.LongTensor(3,72,4).zero_() #包含音素和音调两类
    text_padded[:,:,3] = torch.round(torch.rand(text_padded.shape[:2])*5+10)
    input_lengths = torch.LongTensor([72, 67, 56])
    mel_padded = torch.FloatTensor(3, 80, 1000).zero_()
    max_len = torch.max(input_lengths.data).item()
    output_lengths = torch.LongTensor([800, 900, 1000])
    
    x = (text_padded, input_lengths, mel_padded, max_len, output_lengths)
    mel_out, mel_out_postnet, alignment_out, acoustics_of_phone, join_outs, text_alignment = model(x)
    print(mel_out_postnet.shape, alignment_out.shape)