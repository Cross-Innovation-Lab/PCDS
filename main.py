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

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


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


def main(config):
    dataloader_train, dataloader_test = build_loader(config)
    model = build_model(config)
    model.cuda()
    logger.info(str(model))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    # inputs = torch.randn((1,374,768)).cuda()
    # flops = FlopCountAnalysis(model, inputs)
    # inputs.detach()
    # logger.info(f"number of GFLOPs: {flops.total() / 1e9}")

    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(dataloader_train))

    # if config.MODEL.LABEL_SMOOTHING > 0.:
    #     criterion = LabelSmoothingCrossEntropy(config.MODEL.LABEL_SMOOTHING)
    # else:
    #     criterion = torch.nn.CrossEntropyLoss()

    logger.info('#' * 30 + '  Start Training  ' + '#' * 30)
    Meter = ERCMeter()

    for epoch in range(config.TRAIN.EPOCHS):
        logger.info(f'>> epoch {epoch}')
        train_loss = train_one_epoch(config, model, dataloader_train, optimizer, epoch, lr_scheduler)
        test_loss, fscore, acc, DACC, pred, label = validate(config, dataloader_test, model)
        logger.info(
            f'train loss: {train_loss:.4f}, test loss: {test_loss:.4f}, F1: {fscore}, ACC: {acc}, DACC: {DACC}'
        )
        if Meter.fscore < fscore and config.MODEL.SAVE:
            torch.save(model.state_dict(), f'{config.MODEL.SAVE_PATH}/{config.DATA.DATASET}/pcds-[{config.MODEL.T_S},{config.MODEL.T_I},{config.MODEL.T_E}].pth')
        Meter.update(fscore, acc, DACC, pred, label)
    logger.info('#' * 30 + f'  Summary  ' + '#' * 30)
    logger.info(classification_report(Meter.label, Meter.pred, digits=4))
    logger.info(f'MAX_F1_score: {Meter.fscore}')
    logger.info(f'MAX_ACC_score: {Meter.acc}')
    logger.info(f'MIN_DER: {100.0-Meter.DACC}')


def train_one_epoch(config, model, dataloader, optimizer, epoch, lr_scheduler):
    if epoch < config.TRAIN.EPOCHS_TEACH:
        model.affinity_rnn.clustering = False
    total_loss = 0
    optimizer.zero_grad()
    for idx, (input_audios, input_texts, audio_mask, text_mask, emotions, speakers, _) in enumerate(dataloader):
        model.train()
        outputs = model(input_audios, input_texts, audio_mask, text_mask, emotions, speakers)
        num_steps = len(dataloader)
        loss = outputs.loss
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        # for k, v in model.named_parameters():
        #     print(k,': ',v.grad)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
        optimizer.step()
        lr_scheduler.step_update(epoch * num_steps + idx)
    # lr_scheduler.step()
    if epoch < config.TRAIN.EPOCHS_TEACH:
        model.affinity_rnn.clustering = True
    return total_loss


@torch.no_grad()
def validate(config, data_loader, model):
    model.eval()
    total_loss = 0
    pred_list = []
    label_list = []
    DACC_list = []
    for idx, (input_audios, input_texts, audio_mask, text_mask, emotions, speakers, _) in enumerate(data_loader):
        outputs = model(input_audios, input_texts, audio_mask, text_mask, emotions, speakers)
        loss, logits, pred_speakers, context_mask = outputs.loss, outputs.logits, outputs.pred_speakers, outputs.context_mask
        # measure accuracy and record loss
        total_loss += loss.item()
        pred = list(torch.argmax(logits, 1).detach().cpu().numpy())
        targets = list(emotions[emotions.ne(-100)].detach().cpu().numpy())
        DACC_list.extend(compute_DACC(speakers, pred_speakers, context_mask))
        pred_list.extend(pred)
        label_list.extend(targets)
    fscore = round(f1_score(label_list, pred_list, average='weighted') * 100, 4)
    acc = round(accuracy_score(label_list, pred_list) * 100, 4)
    DACC = round(np.mean(DACC_list) * 100, 4)
    return total_loss, fscore, acc, DACC, pred_list, label_list


if __name__ == '__main__':
    config = parse_option()
    log_file = '{}-{}-{}.log'.format(config.MODEL.NAME, config.DATA.DATASET, strftime("%Y-%m-%d_%H:%M:%S", localtime()))
    logger.addHandler(logging.FileHandler("%s/%s" % (config.LOGPATH, log_file)))
    logger.info('#' * 30 + '  Training Arguments  ' + '#' * 30)
    logger.info(config.dump())
    torch.cuda.set_device(config.LOCAL_RANK)
    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True
    main(config)
