from scipy import optimize
import numpy as np


class ERCMeter:
    """Computes and stores the current best value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.DACC = 0.
        self.acc = 0
        self.fscore = 0.
        self.pred = []
        self.label = []

    def update(self, fscore, acc, DACC, pred, label):
        if fscore > self.fscore:
            self.fscore = fscore
            self.pred = pred
            self.label = label
        if acc > self.acc:
            self.acc = acc
        if DACC > self.DACC:
            self.DACC = DACC


def get_list_inverse_index(unique_ids):
    """Get value to position index from a list of unique ids.

  Args:
    unique_ids: A list of unique integers of strings.

  Returns:
    result: a dict from value to position

  Raises:
    TypeError: If unique_ids is not a list.
  """
    # if not isinstance(unique_ids, list):
    #     raise TypeError('unique_ids must be a list')
    result = dict()
    for i, unique_id in enumerate(unique_ids):
        result[unique_id] = i
    return result


def compute_sequence_match_accuracy(sequence1, sequence2):
    """Compute the accuracy between two sequences by finding optimal matching.

  Args:
    sequence1: A list of integers or strings.
    sequence2: A list of integers or strings.

  Returns:
    accuracy: sequence matching accuracy as a number in [0.0, 1.0]

  Raises:
    TypeError: If sequence1 or sequence2 is not list.
    ValueError: If sequence1 and sequence2 are not same size.
  """
    # if not isinstance(sequence1, list) or not isinstance(sequence2, list):
    #     raise TypeError('sequence1 and sequence2 must be lists')
    # if not sequence1 or len(sequence1) != len(sequence2):
    #     raise ValueError('sequence1 and sequence2 must have the same non-zero length')
    # get unique ids from sequences
    unique_ids1 = sorted(set(sequence1))
    unique_ids2 = sorted(set(sequence2))
    inverse_index1 = get_list_inverse_index(unique_ids1)
    inverse_index2 = get_list_inverse_index(unique_ids2)
    # get the count matrix
    count_matrix = np.zeros((len(unique_ids1), len(unique_ids2)))
    for item1, item2 in zip(sequence1, sequence2):
        index1 = inverse_index1[item1]
        index2 = inverse_index2[item2]
        count_matrix[index1, index2] += 1.0
    row_index, col_index = optimize.linear_sum_assignment(-count_matrix)
    optimal_match_count = count_matrix[row_index, col_index].sum()
    accuracy = optimal_match_count / len(sequence1)
    return accuracy


def compute_DACC(speakers, pred_speakers, context_mask):
    batch_size = speakers.shape[0]
    DACC_list = []
    for i in range(batch_size):
        speaker = speakers[i][context_mask[i]]
        pred_speaker = pred_speakers[i][context_mask[i]]
        DACC_list.append(
            compute_sequence_match_accuracy(speaker.detach().cpu().numpy(),
                                            pred_speaker.detach().cpu().numpy()))
    return DACC_list


# x = np.array([1,1,2,3,4])
# y = np.array([1,1,3,3,4])
# print(compute_sequence_match_accuracy(x,y))