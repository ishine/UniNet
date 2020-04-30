from torch import nn


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, alignment_target, alignments_weights_padded, text_alignment_padded = targets
        mel_target.requires_grad = False
        alignment_target.requires_grad = False

        mel_out, mel_out_postnet, alignment_out, acoustics_of_phone, join_outs, text_alignment = model_output
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)

        gate_loss = nn.BCELoss(reduction='none')(alignment_out, alignment_target).mean(dim=2)
        gate_loss = (gate_loss * alignments_weights_padded).mean()

        join_loss = nn.MSELoss()(acoustics_of_phone, join_outs)

        text_alignment_padded = text_alignment_padded.flatten()
        text_alignment = text_alignment.view(-1, text_alignment.size(2))
        text_alignment_loss = 0.03 * nn.CrossEntropyLoss(ignore_index = -1)(text_alignment, text_alignment_padded)

        return mel_loss + gate_loss, join_loss, text_alignment_loss, mel_loss + gate_loss + join_loss + text_alignment_loss
