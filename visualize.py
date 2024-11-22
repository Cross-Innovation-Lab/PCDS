#!/usr/bin/env python
# coding: utf-8
import argparse
from ast import parse
from secrets import choice
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

from traitlets import default
from loguru import logger
from openTSNE import TSNE

plt.rc('font', family='Times New Roman')

## 函数库

MACOSKO_COLORS = {
    0: "#A5C93D",
    1: "#8B006B",
    2: "#2000D7",
    3: "#538CBA",
    4: "#B33B19",
    5: "#C38A1F",
}
ZEISEL_COLORS = {
    0: "#ff9f2b",
    1: "#3ED3DA",
    2: "#A14D99",
    3: "#FE94B2",
    4: "#98cc41",
    5: "#363AEA",
    6: "#3d672d",
    7: "#9e3d1b",
    8: "#3b1b59",
    9: "#1b5d2f",
    10: "#51bc4c",
    11: "#ffcb9a",
    12: "#768281",
    13: "#a0daaa",
    14: "#8c7d2b",
    15: "#c52d94",
}
MOUSE_10X_COLORS = {
    0: "#FFFF00",
    1: "#1CE6FF",
    2: "#FF34FF",
    3: "#FF4A46",
    4: "#008941",
    5: "#006FA6",
    6: "#A30059",
    7: "#FFDBE5",
    8: "#7A4900",
    9: "#0000A6",
    10: "#63FFAC",
    11: "#B79762",
    12: "#004D43",
    13: "#8FB0FF",
    14: "#997D87",
    15: "#1B4400",
    16: "#809693",
    17: "#5A0007",
    18: "#4FC601",
    19: "#3B5DFF",
    20: "#4A3B53",
    21: "#FF2F80",
    22: "#61615A",
    23: "#BA0900",
    24: "#6B7900",
    25: "#00C2A0",
    26: "#FFAA92",
    27: "#FF90C9",
    28: "#B903AA",
    29: "#D16100",
    30: "#DDEFFF",
    31: "#000035",
    32: "#7B4F4B",
    33: "#A1C299",
    34: "#300018",
    35: "#0AA6D8",
    36: "#013349",
    37: "#00846F",
}


def plot(
    x,
    y1,
    y2,
    ax=None,
    title=None,
    draw_legend=True,
    draw_centers=False,
    draw_cluster_labels=False,
    colors=None,
    legend_kwargs=None,
    label_order=None,
    label_map1=None,
    label_map2=None,
    save_path=None,
    **kwargs
):
    import matplotlib

    if ax is None:
        _, ax = matplotlib.pyplot.subplots(figsize=(16, 16))

    if title is not None:
        ax.set_title(title)

    plot_params = {"alpha": kwargs.get("alpha", 0.8), "s": kwargs.get("s", 32)}

    # Create main plot
    if label_order is not None:
        assert all(np.isin(np.unique(y1), label_order))
        classes1 = [l for l in label_order if l in np.unique(y1)]
        classes2 = [l for l in label_order if l in np.unique(y2)]
    else:
        classes1 = np.unique(y1)
        classes2 = np.unique(y2)
    if colors is None:
        default_colors = matplotlib.rcParams["axes.prop_cycle"]
        colors = {k: v["color"] for k, v in zip(classes1, default_colors())}

    shapes = {0:'o', 1:'^', 2:'D', 3:'*', 4:'X', 5:'p', 6:'h', 7:'P'}

    point_colors = np.array(list(map(colors.get, y1)))
    # point_shapes = np.array(list(map(shapes.get, y2)))

    ax.scatter(x[:, 0], x[:, 1], c=point_colors, rasterized=True, **plot_params)
    # for i in classes2:
    #     index = np.where(y2==i)
    #     ax.scatter(x[index, 0], x[index, 1], c=point_colors[index], marker=shapes[i], **plot_params)

    # Plot mediods
    if draw_centers:
        centers = []
        for yi in classes1:
            mask = yi == y1
            centers.append(np.median(x[mask, :2], axis=0))
        centers = np.array(centers)

        center_colors = list(map(colors.get, classes1))
        ax.scatter(
            centers[:, 0], centers[:, 1], c=center_colors, s=48, alpha=1, edgecolor="k"
        )

        # Draw mediod labels
        if draw_cluster_labels:
            for idx, label in enumerate(classes1):
                ax.text(
                    centers[idx, 0],
                    centers[idx, 1] + 2.2,
                    label,
                    fontsize=kwargs.get("fontsize", 6),
                    horizontalalignment="center",
                )

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([]), ax.axis("off")

    if draw_legend:
        legend_handles = [
            matplotlib.lines.Line2D(
                [],
                [],
                marker="s",
                color="w",
                markerfacecolor=colors[yi],
                ms=32,
                alpha=1,
                linewidth=0,
                label=label_map1[yi],
                markeredgecolor="k",
            )
            for yi in classes1
        ]
        legend_kwargs_ = dict(loc="center left", bbox_to_anchor=(0.9, 0.5), frameon=False, labelspacing=0.2, handletextpad=0.03, prop={'size': 48})
        if legend_kwargs is not None:
            legend_kwargs_.update(legend_kwargs)
        ax.legend(handles=legend_handles, **legend_kwargs_)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)


def feature_tsne(args, X, Y1, Y2, label_map1, label_map2):
    logger.info(f'数据信息: {X.shape}, {Y1.shape}, {Y2.shape}')
    logger.info('正在进行TSNE降维...')
    X_tsne = TSNE(n_components=2, n_jobs=args.n_jobs, perplexity=args.perplexity).fit(X)
    save_path_emo = args.feat_path.replace('.npz', f'_ppl{args.perplexity}_emo.svg')
    save_path_spk = args.feat_path.replace('.npz', f'_ppl{args.perplexity}_spk.svg')
    plot(X_tsne, Y1, Y2, save_path=save_path_emo, colors=MOUSE_10X_COLORS, label_map1=label_map1, label_map2=label_map2)
    plot(X_tsne, Y2, Y1, save_path=save_path_spk, colors=ZEISEL_COLORS, label_map1=label_map2, label_map2=label_map1)
    logger.info(f'TSNE可视化图已保存')


def data2index(args, feats, speaker_labels, emotion_labels):
    if 'IEMOCAP' in args.feat_path:
        speaker_labels = np.array([speaker.split('_')[0] for speaker in speaker_labels])
    speaker_map = {}
    emotion_map = {}
    print(speaker_labels.shape)
    print(emotion_labels.shape)
    print(feats.shape)
    u, _, cnt = np.unique(speaker_labels, return_inverse=True, return_counts=True)
    speaker_freq = dict(zip(u, cnt))
    print(speaker_freq)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='.')
    parser.add_argument('--exam_id', type=str, default='')
    parser.add_argument('--feat_path', type=str, default='visualize/instance_IEMOCAP.npz')
    # parser.add_argument('--feat_type', type=str, choices=['speaker','instance','emotion'], default='speaker')
    parser.add_argument('--n_jobs', type=int, default=4)
    parser.add_argument('--perplexity', type=int, default=30)
    args = parser.parse_args()

    logger.info(f'特征读取：{args.feat_path}')
    data = np.load(args.feat_path, allow_pickle=True)
    feats = data['features']
    speaker_labels = data['speakers']
    emotion_labels = data['emotions']

    # feat_type = args.feat_path.split('/')[-1].split('_')[0]
    speaker_index, speaker_map, emotion_index, emotion_map, feats = data2index(args, feats, speaker_labels, emotion_labels)
    
    # if 'emotion' in args.feat_path:
    #     feature_tsne(args, feats, emotion_index, speaker_index, emotion_map, speaker_map) # emotion
    # else:
    #     feature_tsne(args, feats, speaker_index, emotion_index, speaker_map, emotion_map) # speaker
    feature_tsne(args, feats, emotion_index, speaker_index, emotion_map, speaker_map) # emotion