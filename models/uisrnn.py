import torch
import torch.nn as nn
import numpy as np
from torch import autograd
from torch.nn.utils.rnn import pad_sequence


class BeamState:
    """Structure that contains necessary states for beam search."""
    def __init__(self, source=None):
        if not source:
            self.mean_set = []
            self.hidden_set = []
            self.neg_likelihood = 0
            self.trace = []
            self.block_counts = []
        else:
            self.mean_set = source.mean_set.copy()
            self.hidden_set = source.hidden_set.copy()
            self.trace = source.trace.copy()
            self.block_counts = source.block_counts.copy()
            self.neg_likelihood = source.neg_likelihood

    def append(self, mean, hidden, cluster):
        """Append new item to the BeamState."""
        self.mean_set.append(mean.clone())
        self.hidden_set.append(hidden.clone())
        self.block_counts.append(1)
        self.trace.append(cluster)


class UISRNN(nn.Module):
    def __init__(self, dim, transition_bias=0.5, drop=0.) -> None:
        super().__init__()
        self.device = torch.device('cuda' if (
            torch.cuda.is_available()) else 'cpu')
        self.observation_dim = dim
        self.rnn_model = nn.GRU(input_size=dim,
                                 hidden_size=dim,
                                 num_layers=1,
                                 batch_first=True,
                                 dropout=drop)
        self.mean_head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(),
                                       nn.Linear(dim, dim))
        # self.out_linear = nn.Linear(dim, 2 * dim)
        self.rnn_init_hidden = nn.Parameter(torch.zeros(1, 1, dim))
        # self.rnn_init_cell = nn.Parameter(torch.zeros(1, 1, dim))
        # self.out_rnn = nn.LSTM(input_size=dim,
        #                        hidden_size=dim,
        #                        num_layers=1,
        #                        batch_first=True,
        #                        bidirectional=True,
        #                        dropout=drop)

        sigma2 = 0.1
        self.sigma2 = nn.Parameter(sigma2 * torch.ones(self.observation_dim))
        self.sigma_alpha = 1.0
        self.sigma_beta = 1.0
        self.transition_bias = transition_bias
        self.crp_alpha = 1.0
        self.beam_size = 10
        self.look_ahead = 1
        self.test_iteration = 2

    def _encode_with_rnn(self, rnn, x, length, hidden=None):
        total_length = x.size(1)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(x,
                                                       length.cpu(),
                                                       batch_first=True,
                                                       enforce_sorted=False)
        rnn_outputs, _ = rnn(rnn_inputs, hidden)
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(
            rnn_outputs, batch_first=True, total_length=total_length)
        return rnn_outputs

    def forward(self, x, context_mask, speakers):
        '''
        x: batch_size, seq_num, dim
        context_mask: batch_size, seq_num
        speakers: batch_size, seq_num
        '''
        train_seq = x[context_mask, :]  # [seq_num, dim]
        clusters = speakers[context_mask]  # [seq_num]
        unique_id, seqlens = torch.unique(clusters,
                                          sorted=False,
                                          return_counts=True)
        num_clusters = unique_id.size(0)
        seqlens = torch.add(seqlens, 1)
        max_seq_len = torch.max(seqlens).item()

        # collect input for each cluster
        sub_seq = torch.zeros(num_clusters,
                              max_seq_len,
                              self.observation_dim,
                              device=x.device)
        for i in range(num_clusters):
            sub_seq[i, 1:seqlens[i], :] = train_seq[clusters.eq(
                unique_id[i].item())]
        rnn_truth = sub_seq[:, 1:, :]
        h_0 = self.rnn_init_hidden.repeat(1, num_clusters, 1)
        rnn_output = self._encode_with_rnn(self.rnn_model, sub_seq, seqlens, h_0)
        mean = self.mean_head(rnn_output)  # [num_clusters, max_seq_len, dim]
        # use mean to predict
        mean = torch.cumsum(mean, dim=1)
        mean = torch.reciprocal(
            torch.arange(1, max_seq_len + 1, device=x.device)).view(
                1, max_seq_len, 1) * mean

        # Likelihood part.
        loss1 = weighted_mse_loss(input_tensor=(rnn_truth != 0).float() *
                                  mean[:, :-1, :],
                                  target_tensor=rnn_truth,
                                  weight=1 / (2 * self.sigma2))

        # Sigma2 prior part.
        weight = (((rnn_truth != 0).float() * mean[:, :-1, :] -
                   rnn_truth)**2).view(-1, self.observation_dim)
        num_non_zero = torch.sum((weight != 0).float(), dim=0).squeeze()
        loss2 = sigma2_prior_loss(num_non_zero, self.sigma_alpha,
                                  self.sigma_beta, self.sigma2)

        # Regularization part.
        loss3 = regularization_loss(self.rnn_model.parameters(), 1e-5)
        loss3 += regularization_loss(self.mean_head.parameters(), 1e-5)

        loss = loss1 + loss2 + loss3

        # # collect output with linear output
        # recollect_output = torch.zeros_like(train_seq)  # [seq_num, dim]
        # for i in range(num_clusters):
        #     recollect_output[clusters.eq(unique_id[i].item())] = rnn_output[i, 1:seqlens[i], :]
        # x[context_mask, :] = recollect_output
        # x = self.out_linear(x)
        # return x, loss

        # collect output with rnn output for each speaker
        # rnn_output = self._encode_with_rnn(self.out_rnn, sub_seq, seqlens)
        # recollect_output = torch.zeros(train_seq.size(0),
        #                                self.observation_dim * 2,
        #                                device=x.device)
        # for i in range(num_clusters):
        #     recollect_output[clusters.eq(
        #         unique_id[i].item())] = rnn_output[i, 1:seqlens[i], :]
        # out = torch.zeros(x.size(0),
        #                   x.size(1),
        #                   self.observation_dim * 2,
        #                   device=x.device)
        # out[context_mask, :] = recollect_output
        return speakers, loss

    def _update_beam_state(self, beam_state, look_ahead_seq, cluster_seq):
        """Update a beam state given a look ahead sequence and known cluster
    assignments.

    Args:
      beam_state: A BeamState object.
      look_ahead_seq: Look ahead sequence, size: look_ahead*D.
        look_ahead: number of step to look ahead in the beam search.
        D: observation dimension
      cluster_seq: Cluster assignment sequence for look_ahead_seq.

    Returns:
      new_beam_state: An updated BeamState object.
    """

        loss = 0
        new_beam_state = BeamState(beam_state)
        for sub_idx, cluster in enumerate(cluster_seq):
            if cluster > len(new_beam_state.mean_set):  # invalid trace
                new_beam_state.neg_likelihood = float('inf')
                break
            elif cluster < len(new_beam_state.mean_set):  # existing cluster
                last_cluster = new_beam_state.trace[-1]
                loss = weighted_mse_loss(
                    input_tensor=torch.squeeze(
                        new_beam_state.mean_set[cluster]),
                    target_tensor=look_ahead_seq[sub_idx, :],
                    weight=1 / (2 * self.sigma2)).cpu().detach().numpy()
                if cluster == last_cluster:
                    loss -= np.log(1 - self.transition_bias)
                else:
                    loss -= np.log(self.transition_bias) + np.log(
                        new_beam_state.block_counts[cluster]) - np.log(
                            sum(new_beam_state.block_counts) + self.crp_alpha)
                # update new mean and new hidden
                mean, hidden = self.rnn_model(
                    look_ahead_seq[sub_idx, :].unsqueeze(0).unsqueeze(0),
                    new_beam_state.hidden_set[cluster])
                mean = self.mean_head(mean)
                new_beam_state.mean_set[cluster] = (
                    new_beam_state.mean_set[cluster] *
                    ((np.array(new_beam_state.trace) == cluster).sum() -
                     1).astype(float) + mean.clone()) / (np.array(
                         new_beam_state.trace) == cluster).sum().astype(
                             float)  # use mean to predict
                new_beam_state.hidden_set[cluster] = hidden.clone()
                if cluster != last_cluster:
                    new_beam_state.block_counts[cluster] += 1
                new_beam_state.trace.append(cluster)
            else:  # new cluster
                init_input = autograd.Variable(
                    torch.zeros(
                        self.observation_dim)).unsqueeze(0).unsqueeze(0).to(
                            self.device)
                mean, hidden = self.rnn_model(init_input, self.rnn_init_hidden)
                mean = self.mean_head(mean)
                loss = weighted_mse_loss(
                    input_tensor=torch.squeeze(mean),
                    target_tensor=look_ahead_seq[sub_idx, :],
                    weight=1 / (2 * self.sigma2)).cpu().detach().numpy()
                loss -= np.log(self.transition_bias) + np.log(
                    self.crp_alpha) - np.log(
                        sum(new_beam_state.block_counts) + self.crp_alpha)
                # update new min and new hidden
                mean, hidden = self.rnn_model(
                    look_ahead_seq[sub_idx, :].unsqueeze(0).unsqueeze(0),
                    hidden)
                mean = self.mean_head(mean)
                new_beam_state.append(mean, hidden, cluster)
            new_beam_state.neg_likelihood += loss
        return new_beam_state

    def _calculate_score(self, beam_state, look_ahead_seq):
        """Calculate negative log likelihoods for all possible state allocations
       of a look ahead sequence, according to the current beam state.

    Args:
      beam_state: A BeamState object.
      look_ahead_seq: Look ahead sequence, size: look_ahead*D.
        look_ahead: number of step to look ahead in the beam search.
        D: observation dimension

    Returns:
      beam_score_set: a set of scores for each possible state allocation.
    """

        look_ahead, _ = look_ahead_seq.shape
        beam_num_clusters = len(beam_state.mean_set)
        beam_score_set = float('inf') * np.ones(beam_num_clusters + 1 +
                                                np.arange(look_ahead))
        for cluster_seq, _ in np.ndenumerate(beam_score_set):
            updated_beam_state = self._update_beam_state(
                beam_state, look_ahead_seq, cluster_seq)
            beam_score_set[cluster_seq] = updated_beam_state.neg_likelihood
        return beam_score_set

    def predict_single(self, test_sequence):
        """Predict labels for a single test sequence using UISRNN model.

    Args:
      test_sequence: the test observation sequence, which is 2-dim numpy array
        of real numbers, of size `N * D`.

        - `N`: length of one test utterance.
        - `D` : observation dimension.

        For example:
      ```
      test_sequence =
      [[2.2 -1.0 3.0 5.6]    --> 1st entry of utterance 'iccc'
       [0.5 1.8 -3.2 0.4]    --> 2nd entry of utterance 'iccc'
       [-2.2 5.0 1.8 3.7]    --> 3rd entry of utterance 'iccc'
       [-3.8 0.1 1.4 3.3]    --> 4th entry of utterance 'iccc'
       [0.1 2.7 3.5 -1.7]]   --> 5th entry of utterance 'iccc'
      ```
        Here `N=5`, `D=4`.
      args: Inference configurations. See `arguments.py` for details.

    Returns:
      predicted_cluster_id: predicted speaker id sequence, which is
        an array of integers, of size `N`.
        For example, `predicted_cluster_id = [0, 1, 0, 0, 1]`

    Raises:
      TypeError: If test_sequence is of wrong type.
      ValueError: If test_sequence has wrong dimension.
    """
        # check size
        test_sequence_length, observation_dim = test_sequence.shape
        if observation_dim != self.observation_dim:
            raise ValueError(
                'test_sequence does not match the dimension specified '
                'by args.observation_dim.')

        self.rnn_model.eval()
        test_sequence = np.tile(test_sequence, (self.test_iteration, 1))
        test_sequence = autograd.Variable(
            torch.from_numpy(test_sequence).float()).to(self.device)
        # bookkeeping for beam search
        beam_set = [BeamState()]
        for num_iter in np.arange(0,
                                  self.test_iteration * test_sequence_length,
                                  self.look_ahead):
            max_clusters = max(
                [len(beam_state.mean_set) for beam_state in beam_set])
            look_ahead_seq = test_sequence[num_iter:num_iter +
                                           self.look_ahead, :]
            look_ahead_seq_length = look_ahead_seq.shape[0]
            score_set = float('inf') * np.ones(
                np.append(self.beam_size,
                          max_clusters + 1 + np.arange(look_ahead_seq_length)))
            for beam_rank, beam_state in enumerate(beam_set):
                beam_score_set = self._calculate_score(beam_state,
                                                       look_ahead_seq)
                score_set[beam_rank, :] = np.pad(
                    beam_score_set,
                    np.tile([[0, max_clusters - len(beam_state.mean_set)]],
                            (look_ahead_seq_length, 1)),
                    'constant',
                    constant_values=float('inf'))
            # find top scores
            score_ranked = np.sort(score_set, axis=None)
            score_ranked[score_ranked == float('inf')] = 0
            score_ranked = np.trim_zeros(score_ranked)
            idx_ranked = np.argsort(score_set, axis=None)
            updated_beam_set = []
            for new_beam_rank in range(
                    np.min((len(score_ranked), self.beam_size))):
                total_idx = np.unravel_index(idx_ranked[new_beam_rank],
                                             score_set.shape)
                prev_beam_rank = total_idx[0].item()
                cluster_seq = total_idx[1:]
                updated_beam_state = self._update_beam_state(
                    beam_set[prev_beam_rank], look_ahead_seq, cluster_seq)
                updated_beam_set.append(updated_beam_state)
            beam_set = updated_beam_set
        predicted_cluster_id = beam_set[0].trace[-test_sequence_length:]
        return np.array(predicted_cluster_id)

    def predict(self, x, context_mask):
        """
        x: batch_size, seq_num, dim
        context_mask: batch_size, seq_num
        """
        batch_size, seq_num = context_mask.shape
        loss_affinity = 0
        pred_speakers = torch.zeros_like(context_mask, dtype=torch.long, device=x.device)
        for i in range(batch_size):
            sequence = x[i][context_mask[i]]
            clusters = self.predict_single(sequence.cpu().detach().numpy())
            pred_speakers[i][context_mask[i]] = torch.from_numpy(
                clusters).long().to(context_mask.device)

        return pred_speakers, loss_affinity


def weighted_mse_loss(input_tensor, target_tensor, weight=1):
    """Compute weighted MSE loss.

  Note that we are doing weighted loss that only sum up over non-zero entries.

  Args:
    input_tensor: input tensor
    target_tensor: target tensor
    weight: weight tensor, in this case 1/sigma^2

  Returns:
    the weighted MSE loss
  """
    observation_dim = input_tensor.size()[-1]
    streched_tensor = ((input_tensor - target_tensor)**2).view(
        -1, observation_dim)
    entry_num = float(streched_tensor.size()[0])
    non_zero_entry_num = torch.sum(streched_tensor[:, 0] != 0).float()
    weighted_tensor = torch.mm(
        ((input_tensor - target_tensor)**2).view(-1, observation_dim),
        (torch.diag(weight.float().view(-1))))
    return torch.mean(
        weighted_tensor) * weight.nelement() * entry_num / non_zero_entry_num


def sigma2_prior_loss(num_non_zero, sigma_alpha, sigma_beta, sigma2):
    """Compute sigma2 prior loss.

  Args:
    num_non_zero: since rnn_truth is a collection of different length sequences
        padded with zeros to fit them into a tensor, we count the sum of
        'real lengths' of all sequences
    sigma_alpha: inverse gamma shape
    sigma_beta: inverse gamma scale
    sigma2: sigma squared

  Returns:
    the sigma2 prior loss
  """
    return ((2 * sigma_alpha + num_non_zero + 2) / (2 * num_non_zero) *
            torch.log(sigma2)).sum() + (sigma_beta /
                                        (sigma2 * num_non_zero)).sum()


def regularization_loss(params, weight):
    """Compute regularization loss.

  Args:
    params: iterable of all parameters
    weight: weight for the regularization term

  Returns:
    the regularization loss
  """
    l2_reg = 0
    for param in params:
        l2_reg += torch.norm(param)
    return weight * l2_reg

