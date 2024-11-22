import torch
import torch.nn as nn
import torch.nn.functional as F
from .clustering import SpectralClusterer
from .loss import SupConLoss


class RNN(nn.Module):

    def __init__(self,
                 dim,
                 length,
                 num_classes=4,
                 shift=False,
                 stride=1,
                 n_div=1,
                 bidirectional=True,
                 mask=False,
                 drop=0.):
        super().__init__()
        # self.augment = Specaugment(dim)

        self.rnn = nn.LSTM(input_size=dim,
                           hidden_size=dim,
                           num_layers=1,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop)

        self.norm = nn.LayerNorm(dim * 2)
        self.head = nn.Linear(dim * 2, num_classes)

    def encode_with_rnn(self, x, length):
        total_length = x.size(1)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(x, length.cpu(), batch_first=True, enforce_sorted=False)
        rnn_outputs, _ = self.rnn(rnn_inputs)
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True, total_length=total_length)
        return rnn_outputs

    def forward(self, x, context_mask):
        """
        x: batch_size, seq_num, channel
        context_mask: batch_size, seq_num

        """
        seqlens = torch.sum(context_mask, dim=-1)
        # x = self.augment(x)
        rnn_output = self.encode_with_rnn(x, seqlens)
        # conv_output = self.conv(x_aug).transpose(-2, -1)

        # if self.use_rho:
        #     conv_output = torch.sigmoid(self.rho) * conv_output
        # # rnn_output, _ = self.rnn2(x_aug)
        # # x = self.linear(x)
        # x = self.head(torch.mean(self.norm(rnn_output), dim=1))

        # x = self.head(torch.mean(self.norm(rnn_output+conv_output), dim=1))

        x = self.head(self.norm(rnn_output))
        return x


class SpeakerRNN(nn.Module):

    def __init__(self, dim, drop=0.) -> None:
        super().__init__()
        self.observation_dim = dim
        self.rnn = nn.LSTM(input_size=dim,
                           hidden_size=dim // 2,
                           num_layers=1,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop)

    def _encode_with_rnn(self, rnn, x, length, hidden=None):
        total_length = x.size(1)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(x, length.cpu(), batch_first=True, enforce_sorted=False)
        rnn_outputs, _ = rnn(rnn_inputs, hidden)
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True, total_length=total_length)
        return rnn_outputs

    def forward(self, x, context_mask, speakers):
        '''
        x: batch_size, seq_num, dim
        context_mask: batch_size, seq_num
        speakers: batch_size, seq_num
        '''
        train_seq = x[context_mask, :]  # [seq_num, dim]
        clusters = speakers[context_mask]  # [seq_num]
        unique_id, seqlens = torch.unique(clusters, sorted=False, return_counts=True)
        num_clusters = unique_id.size(0)
        max_seq_len = torch.max(seqlens).item()

        # collect input for each cluster
        sub_seq = torch.zeros(num_clusters, max_seq_len, self.observation_dim, device=x.device)
        for i in range(num_clusters):
            sub_seq[i, :seqlens[i], :] = train_seq[clusters.eq(unique_id[i].item())]
        rnn_output = self._encode_with_rnn(self.rnn, sub_seq, seqlens)

        # collect output with rnn output for each speaker

        recollect_output = torch.zeros(train_seq.size(0), rnn_output.size(-1), device=x.device)
        for i in range(num_clusters):
            recollect_output[clusters.eq(unique_id[i].item())] = rnn_output[i, :seqlens[i], :]
        out = torch.zeros(x.size(0), x.size(1), rnn_output.size(-1), device=x.device)
        out[context_mask, :] = recollect_output
        return out


class Clusterer(nn.Module):

    def __init__(self, dim, drop=0.) -> None:
        super().__init__()
        # self.project = nn.Linear(dim, dim // 2)
        # self.affrnn = nn.LSTM(input_size=dim,
        #                       hidden_size=dim // 2,
        #                       num_layers=1,
        #                       batch_first=True,
        #                       bidirectional=True,
        #                       dropout=drop)

        self.affrnn = nn.LSTM(input_size=dim,
                              hidden_size=dim // 2,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=True,
                              dropout=drop)

        # self.norm = nn.LayerNorm(2*dim)
        # self.sim_head = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, 1), nn.Sigmoid())
        self.sim_head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim // 2))
        # self.speaker_rnn = SpeakerRNN(dim)
        self.clustering = True
        if self.clustering:
            self.clusterer = SpectralClusterer()

    def _encode_with_rnn(self, rnn, x, length):
        total_length = x.size(1)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(x, length.cpu(), batch_first=True, enforce_sorted=False)
        rnn_outputs, _ = rnn(rnn_inputs)
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True, total_length=total_length)
        return rnn_outputs

    def forward(self, x, context_mask, speakers, x_aug=None):
        '''
        x: batch_size, seq_num, dim
        context_mask: batch_size, seq_num
        speakers: batch_size, seq_num
        '''
        batch_size, seq_num, dim = x.shape
        seqlens = torch.sum(context_mask, dim=-1)
        criterion = nn.BCELoss()
        matrices = []
        labels = []
        unique_idx = 0

        if self.clustering:
            pred_speakers = torch.zeros_like(speakers, device=speakers.device)
        else:
            pred_speakers = None

        x = self._encode_with_rnn(self.affrnn, x, seqlens)
        if x_aug is not None:
            x_aug = self._encode_with_rnn(self.affrnn, x_aug, seqlens)
        for i in range(batch_size):
            sequence = x[i][context_mask[i]]  # [seq_num', dim]
            speaker = speakers[i][context_mask[i]]
            sequence = self.sim_head(sequence)
            if x_aug is not None:
                sequence_aug = x_aug[i][context_mask[i]]
                sequence_aug = self.sim_head(sequence_aug)
                affinity_matrix = torch.sigmoid((torch.matmul(sequence, sequence_aug.T)))
            else:
                affinity_matrix = torch.sigmoid((torch.matmul(sequence, sequence.T)))
            label_matrix = torch.eq(speaker.unsqueeze(0), speaker.unsqueeze(-1)).float()
            matrices.append(affinity_matrix.flatten())
            labels.append(label_matrix.flatten())
            if self.clustering:
                clusters, num_unique = self.clusterer.predict(affinity_matrix.clone().detach().cpu().numpy())
                clusters += unique_idx
                pred_speakers[i][context_mask[i]] = torch.from_numpy(clusters).long().to(speakers.device)
                unique_idx += num_unique

        loss = criterion(torch.cat(matrices), torch.cat(labels))

        return pred_speakers, loss


class CClusterer(nn.Module):

    def __init__(self, dim, T=1.0, drop=0.) -> None:
        super().__init__()
        # self.project = nn.Linear(dim, dim // 2)
        # self.affrnn = nn.LSTM(input_size=dim,
        #                       hidden_size=dim // 2,
        #                       num_layers=1,
        #                       batch_first=True,
        #                       bidirectional=True,
        #                       dropout=drop)

        self.affrnn = nn.LSTM(input_size=dim,
                              hidden_size=dim // 2,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=True,
                              dropout=drop)

        # self.norm = nn.LayerNorm(2*dim)
        # self.sim_head = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, 1), nn.Sigmoid())
        self.sim_head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim // 2))
        # self.speaker_rnn = SpeakerRNN(dim)
        self.criterion = SupConLoss(temperature=T)
        # self.bce = nn.BCELoss()
        self.clustering = True
        if self.clustering:
            self.clusterer = SpectralClusterer()

    def _encode_with_rnn(self, rnn, x, length):
        total_length = x.size(1)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(x, length.cpu(), batch_first=True, enforce_sorted=False)
        rnn_outputs, _ = rnn(rnn_inputs)
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True, total_length=total_length)
        return rnn_outputs

    def forward(self, x, context_mask, speakers, x_aug=None):
        '''
        x: batch_size, seq_num, dim
        context_mask: batch_size, seq_num
        speakers: batch_size, seq_num
        '''
        batch_size, seq_num, dim = x.shape
        seqlens = torch.sum(context_mask, dim=-1)
        loss = 0
        unique_idx = 0
        matrices = []
        labels = []

        if self.clustering:
            pred_speakers = torch.zeros_like(speakers, device=speakers.device)
        else:
            pred_speakers = None

        x = self._encode_with_rnn(self.affrnn, x, seqlens)
        if x_aug is not None:
            x_aug = self._encode_with_rnn(self.affrnn, x_aug, seqlens)
        for i in range(batch_size):
            sequence = x[i][context_mask[i]]  # [seq_num', dim]
            speaker = speakers[i][context_mask[i]]
            sequence = self.sim_head(sequence)
            sequence_normalized = F.normalize(sequence, dim=-1)
            if x_aug is not None:
                sequence_aug = x_aug[i][context_mask[i]]
                sequence_aug_normalized = F.normalize(self.sim_head(sequence_aug), dim=-1)
                loss += self.criterion(torch.stack((sequence_normalized, sequence_aug_normalized), dim=1), speaker)
                affinity_matrix = torch.sigmoid((torch.matmul(sequence_normalized, sequence_aug_normalized.T)))
                # affinity_matrix = torch.div(torch.matmul(sequence_normalized, sequence_aug_normalized.T) + 1.0, 20)
                
            else:
                affinity_matrix = torch.div(torch.matmul(sequence_normalized, sequence_normalized.T) + 1.0, 2.0)
                loss += self.criterion(torch.stack((sequence_normalized, sequence_normalized), dim=1), speaker)
            # label_matrix = torch.eq(speaker.unsqueeze(0), speaker.unsqueeze(-1)).float()
            # matrices.append(affinity_matrix.flatten())
            # labels.append(label_matrix.flatten())
            if self.clustering:
                clusters, num_unique = self.clusterer.predict(affinity_matrix.clone().detach().cpu().numpy())
                clusters += unique_idx
                pred_speakers[i][context_mask[i]] = torch.from_numpy(clusters).long().to(speakers.device)
                unique_idx += num_unique
        
        # loss += self.bce(torch.cat(matrices), torch.cat(labels))

        return pred_speakers, loss
