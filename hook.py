from cProfile import label
import os
import sys
import logging
import argparse
import numpy as np
from time import strftime, localtime
from sklearn.metrics import f1_score, accuracy_score, classification_report

import torch
import torch.backends.cudnn as cudnn
from timm.loss import LabelSmoothingCrossEntropy
# from fvcore.nn import FlopCountAnalysis

from config import get_config
from models import build_model
from dataset import build_loader
from optimizer import build_optimizer
from lr_scheduler import build_scheduler
from metrics import ERCMeter, compute_DACC

speaker_feat = []
speaker_label = []
instance_feat = []
emotion_feat = []
emotion_label = []

def parse_option():
    parser = argparse.ArgumentParser('PCDS', add_help=False)

    # # easy config modification
    # parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    # parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--batchsize', type=int, help="batch size for single GPU")
    parser.add_argument('--logpath', type=str, help='directory path to log')
    parser.add_argument('--logname', type=str, help='name of log')
    parser.add_argument('--dataset', type=str, help='dataset for training/testing')
    parser.add_argument('--shift', action='store_true', help='whether to use temporal shift')
    parser.add_argument('--stride', type=int, help='temporal shift stride')
    parser.add_argument('--ndiv', type=int, help='temporal shift portion')
    parser.add_argument('--k', type=int, help='queue size')
    parser.add_argument('--ts', type=float, help='temperature of speaker contrast')
    parser.add_argument('--ti', type=float, help='temperature of instance contrast')
    parser.add_argument('--te', type=float, help='temperature of emotion contrast')
    parser.add_argument('--bidirectional', action='store_true', help='temporal shift direction')
    parser.add_argument('--kernel',
                        type=int,
                        help='kernel size of first convolution layer (only work for model type CNN)')
    parser.add_argument('--gpu', type=str, help='gpu rank to use')
    parser.add_argument('--seed', type=int, help='seed')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return config

def forward_hook_speaker(module, input, output):
    global context_mask
    x = input[0].detach()
    context_mask = input[1].detach()
    speaker_feat.append(x[context_mask].detach().cpu().numpy())


def forward_hook_instance(module, input, output):
    instance_feat.append(input[0][context_mask].detach().cpu().numpy())

def forward_hook_emotion(module, input, output):
    emotion_feat.append(input[0][context_mask].detach().cpu().numpy())

def main(config):
    dataloader_train, dataloader_test = build_loader(config)
    model = build_model(config)
    model.load_state_dict(torch.load(f'{config.MODEL.SAVE_PATH}/{config.DATA.DATASET}/pcds-[{config.MODEL.T_S},{config.MODEL.T_I},{config.MODEL.T_E}].pth'))
    model.eval()
    model.cuda()
    model.clusterer.register_forward_hook(forward_hook_speaker)
    model.ic_head_text.register_forward_hook(forward_hook_instance)
    model.head.register_forward_hook(forward_hook_emotion)
    
    for idx, (input_audios, input_texts, audio_mask, text_mask, emotions, speakers, speakers_str) in enumerate(dataloader_test):
        speaker_label.extend(speakers_str)
        outputs = model(input_audios, input_texts, audio_mask, text_mask, emotions, speakers)
        emotion_label.extend(emotions[outputs.context_mask].detach().cpu().numpy())

    np.savez(f'visualize/speaker_{config.DATA.DATASET}.npz', features=np.concatenate(speaker_feat, axis=0), speakers=speaker_label, emotions=emotion_label)
    np.savez(f'visualize/emotion_{config.DATA.DATASET}.npz', features=np.concatenate(emotion_feat, axis=0), speakers=speaker_label, emotions=emotion_label)
    np.savez(f'visualize/instance_{config.DATA.DATASET}.npz', features=np.concatenate(instance_feat, axis=0), speakers=speaker_label, emotions=emotion_label)


if __name__ == '__main__':
    config = parse_option()
    torch.cuda.set_device(config.LOCAL_RANK)
    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True
    main(config)
