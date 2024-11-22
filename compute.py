import argparse
from ast import parse
from secrets import choice
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist


def data2index(args, feats, speaker_labels, emotion_labels):
    if 'IEMOCAP' in args.feat_path:
        speaker_labels = np.array([speaker.split('_')[0] for speaker in speaker_labels])
    speaker_map = {}
    emotion_map = {}
    u, _, cnt = np.unique(speaker_labels, return_inverse=True, return_counts=True)
    speaker_freq = dict(zip(u, cnt))
    label_cnt = np.array(list(map(lambda x: speaker_freq[x], speaker_labels)))
    labels_mask = np.where(label_cnt>=100)
    u, speaker_index = np.unique(speaker_labels[labels_mask], return_inverse=True)
    feats = feats[labels_mask]
    emotion_index = emotion_labels[labels_mask]
    for i, speaker in enumerate(u):
        speaker_map[i] = speaker
    
    if 'MELD' in args.feat_path:
        emotion_map = {0: 'neutral', 1: 'surprise', 2: 'fear', 3: 'sadness', 4: 'joy', 5: 'disgust', 6: 'anger'}
    elif 'IEMOCAP' in args.feat_path:
        emotion_map = {0: 'happy', 1: 'sad', 2: 'neutral', 3: 'angry', 4: 'excited', 5: 'frustrated'}

    return speaker_index, speaker_map, emotion_index, emotion_map, feats

def norm(x, eps=1e-8):
    x_norm = np.linalg.norm(x, axis=-1, keepdims=True)
    x_eps = np.ones_like(x_norm) * eps
    return x / np.maximum(x_norm, x_eps)

def lalign(x, labels):
    classes = np.unique(labels)
    dist_list = []
    for i in classes:
        index = np.where(labels==i)
        x_class = x[index]
        dist_mean = np.power(pdist(x_class), 2).mean()
        dist_list.append(dist_mean)
    return np.mean(dist_list)

def lunif(x, t=2):
    dist = np.power(pdist(x), 2)
    return np.log(np.exp(dist*-t).mean())


def compute(feats, emotion_label, speaker_label):
    feats = norm(feats)
    emo_align = lalign(feats, emotion_label)
    spk_align = lalign(feats, speaker_label)
    unif = lunif(feats)
    print(f'emotion alignment: {emo_align}')
    print(f'speaker alignment: {spk_align}')
    print(f'uniformity: {unif}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_path', type=str, default='visualize/emotion_IEMOCAP.npz')
    # parser.add_argument('--feat_type', type=str, choices=['speaker','instance','emotion'], default='speaker')
    parser.add_argument('--n_jobs', type=int, default=4)
    args = parser.parse_args()

    data = np.load(args.feat_path, allow_pickle=True)
    feats = data['features']
    speaker_labels = data['speakers']
    emotion_labels = data['emotions']

    # feat_type = args.feat_path.split('/')[-1].split('_')[0]
    speaker_index, speaker_map, emotion_index, emotion_map, feats = data2index(args, feats, speaker_labels, emotion_labels)
    compute(feats, emotion_index, speaker_index)
