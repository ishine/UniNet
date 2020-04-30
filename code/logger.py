import random
import torch
from tensorboardX import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy, plot_weight_outputs_to_numpy


class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir)

    def log_training(self, reduced_loss_main, reduced_loss_join, reduced_loss_class, reduced_loss,
                     annealing_value, grad_norm, learning_rate, duration,
                     iteration):
            if annealing_value != None:
                self.add_scalar("training.P(teacher_forcing)", annealing_value, iteration)
            self.add_scalar("training.loss_main", reduced_loss_main, iteration)
            self.add_scalar("training.loss_join", reduced_loss_join, iteration)
            self.add_scalar("training.loss_class", reduced_loss_class, iteration)
            self.add_scalar("training.loss", reduced_loss, iteration)
            self.add_scalar("grad.norm", grad_norm, iteration)
            self.add_scalar("learning.rate", learning_rate, iteration)
            self.add_scalar("duration", duration, iteration)

    def log_validation(self, reduced_loss_main, reduced_loss_join, reduced_loss_class, reduced_loss, model, y, y_pred, iteration):
        self.add_scalar("validation.loss_mian", reduced_loss_main, iteration)
        self.add_scalar("validation.loss_join", reduced_loss_join, iteration)
        self.add_scalar("validation.loss_class", reduced_loss_class, iteration)

        self.add_scalar("validation.loss", reduced_loss, iteration)
        _, mel_outputs, alignment_outputs, acoustics_of_phone, join_outs, text_alignment = y_pred
        mel_targets, alignment_targets, alignments_weights, text_alignment_padded = y

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, mel_targets.size(0) - 1)
        self.add_image(
            "alignment_target",
            plot_alignment_to_numpy(alignment_targets[idx].data.cpu().numpy().T),
            iteration)
        self.add_image(
            "alignment_output",
            plot_alignment_to_numpy(alignment_outputs[idx].data.cpu().numpy().T),
            iteration)
        self.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            iteration)
        self.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            iteration)
        self.add_image(
            "alignments_weights",
            plot_weight_outputs_to_numpy(
                alignments_weights[idx].data.cpu().numpy()),
            iteration)
        self.add_image(
            "acoustic_of_phone_reference",
            plot_alignment_to_numpy(acoustics_of_phone[idx].data.cpu().numpy().T,
                    x_label = 'Encoder timestep', y_label = 'Phoneme Acoustic'),
            iteration)
        self.add_image(
            "acoustic_of_phone_predicted",
            plot_alignment_to_numpy(join_outs[idx].data.cpu().numpy().T,
                    x_label = 'Encoder timestep', y_label = 'Phoneme Acoustic'),
            iteration)
        self.add_image(
            "phone_level_acoustic_text_alignment_output",
            plot_alignment_to_numpy(text_alignment[idx].data.cpu().numpy().T,
                    figsize = (8, 6),
                    x_label = 'Encoder timestep', y_label = 'Encoder timestep'),
            iteration)