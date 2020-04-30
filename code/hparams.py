import tensorflow as tf
from symbols import phone_set, tone2id


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=200,
        iters_per_checkpoint=180,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:12345",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=['embedding_phoneme.weight, embedding_tone.weight', 'embedding_RPB.weight'],
        more_information = False,
        decoder_training_mode = 'teacher forcing', # teacher forcing  Or  random annealing

        ################################
        # Data Parameters              #
        ################################
        load_mel_from_disk=True,
        training_lst='../filelists/train_file.lst',
        validation_lst='../filelists/val_file.lst',
        audio_path='../data/audio',
        mel_path='../data/mel_15ms',
        lab_path='../data/text_addtime',
        MelStd_mel ='../data/MeanStd_Tacotron_mel_15ms.npy',

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=16000,
        filter_length=1024,
        hop_length=240,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=None,  # if None, half the sampling rate

        ################################
        # Model Parameters             #
        ################################
        n_symbols_phoneme=61,
        symbols_embedding_dim_phoneme=206,
        n_symbols_tone=8,
        symbols_embedding_dim_tone=25,
        n_symbols_RPB=11,   # RPB == Rhythm_phrase_boundary
        symbols_embedding_dim_RPB=25,

        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=256,

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        max_decoder_steps = 30000,
        prenet_dim=128,
        p_decoder_dropout=0.1,

        frame_level_rnn_dim=256,
        self_attention_dim = 512,
        phone_level_rnn_dim=128,
        join_model_hidden_dim=256,
        decoder_rnn_dim=512,

        # Mel-post processing network parameters
        postnet_embedding_dim=256,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        # Attention parameters
        attention_dim=128,
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-3,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=40,
        mask_padding=True  # set model's padded outputs to padded values
    )

    if hparams_string:
        tf.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams
