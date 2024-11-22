import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch import autograd
from torch.nn.utils.rnn import pad_sequence
from itertools import permutations


class EncoderDecoderAttractor(nn.Module):
    def __init__(self, dim, max_speaker_num=3, threshold=0.5, drop=0.) -> None:
        super().__init__()
        self.encoder = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=1, batch_first=True, dropout=drop)
        self.decoder = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=1, batch_first=True, dropout=drop)
        self.counter = nn.Sequential(nn.Linear(dim, 1), torch.nn.Sigmoid())
        self.max_speaker_num = max_speaker_num
        self.threshold = threshold
        self.crterion_att = nn.BCELoss()

    def forward(self, x, context_mask, speakers, shuffle=True):
        '''
        x: batch_size, seq_num, dim
        context_mask: batch_size, seq_num
        speakers: batch_size, seq_num
        '''
        batch_size, seq_num, dim = x.shape
        attractor_loss = 0
        PIT_loss = 0
        pred_speakers = torch.zeros_like(speakers)
        for i in range(batch_size):
            sequence = x[i][context_mask[i]]
            seq_len = sequence.size(0)
            speaker = speakers[i][context_mask[i]]
            unique_id, speaker_label = torch.unique(speaker, sorted=False, return_inverse=True)
            num_speaker = unique_id.size(0)
            speaker_label = F.one_hot(speaker_label, num_classes=num_speaker).float()
            if shuffle:
                index = torch.randperm(seq_len, device=sequence.device)
                sequence_perm = sequence[index]
                speaker_label_perm = speaker_label[index]
            else:
                index = torch.arange(seq_len)
                sequence_perm = sequence
                speaker_label_perm = speaker_label

            _, h_c = self.encoder(sequence_perm)
            if self.training:
                decoder_input = torch.zeros(num_speaker + 1, dim, device=sequence.device)
                attractors, _ = self.decoder(decoder_input, h_c)
                attractor_label = torch.from_numpy(np.array([1] * num_speaker + [0])).unsqueeze(-1).float().to(sequence.device)
                attractor_loss += self.crterion_att(self.counter(attractors), attractor_label)
                pred = torch.matmul(sequence_perm, attractors[:-1,:].T)  # [seqlen, num_speaker]
                PIT_loss += pit_loss(pred, speaker_label_perm)
                pred_speaker = torch.zeros_like(speaker)
                pred_speaker[index] = torch.argmax(pred.softmax(dim=-1), dim=-1)
                pred_speakers[i][context_mask[i]] = pred_speaker
            else:
                decoder_input = torch.zeros(self.max_speaker_num, dim, device=sequence.device)
                attractors, _ = self.decoder(decoder_input, h_c)
                probs = self.counter(attractors)
                silences = torch.arange(self.max_speaker_num, device=sequence.device)[probs.squeeze().lt(self.threshold)]
                silence = silences[0].item() if silences.numel() else self.max_speaker_num
                silence = silence if silence else silence + 1
                pred = torch.matmul(sequence_perm, attractors[:silence, :].T).softmax(dim=-1)
                pred_speaker = torch.zeros_like(speaker)
                pred_speaker[index] = torch.argmax(pred, dim=-1)
                pred_speakers[i][context_mask[i]] = pred_speaker
        return pred_speakers, PIT_loss + attractor_loss


def pit_loss(pred, label, label_delay=0):
    """
    Permutation-invariant training (PIT) cross entropy loss function.
    Args:
      pred:  (T,C)-shaped pre-activation values
      label: (T,C)-shaped labels in {0,1}
      label_delay: if label_delay == 5:
            pred: 0 1 2 3 4 | 5 6 ... 99 100 |
          label: x x x x x | 0 1 ... 94  95 | 96 97 98 99 100
          calculated area: | <------------> |
    Returns:
      min_loss: (1,)-shape mean cross entropy
      label_perms[min_index]: permutated labels
    """
    # label permutations along the speaker axis
    criterion = nn.CrossEntropyLoss()
    label_perms = [label[..., list(p)] for p in permutations(range(label.shape[-1]))]
    losses = torch.stack([criterion(pred[label_delay:, ...], l[:len(l) - label_delay, ...]) for l in label_perms])
    min_loss = losses.min() * (len(label) - label_delay)
    min_index = losses.argmin().detach()

    # return min_loss, label_perms[min_index]
    return min_loss
