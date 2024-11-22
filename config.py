import os
import yaml
from pickle import FALSE
from yacs.config import CfgNode as CN

dataset = {
    'IEMOCAP': {
        'data_path': './dataset/IEMOCAP/iemocap_feature.pkl',
        'log_path': './log/IEMOCAP/baseline',
        'audio_max_length': 120000,
        'num_classes': 6
    },
    'MELD': {
        'data_path': './dataset/MELD/meld_feature.pkl',
        'log_path': './log/MELD/baseline',
        'audio_max_length': 56000,
        'num_classes': 7
    }
}

_C = CN()

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
# logpath :['[gru,conv1d]', '[rnn,conv1d]','rnn-2layer']
_C.CFGFILE = './configs'
_C.LOGPATH = './log/IEMOCAP/baseline'
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 2
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = './dataset/IEMOCAP/iemocap_feature.pkl'
# Dataset name
_C.DATA.DATASET = 'MELD'
# Feature augmentation
_C.DATA.SPEAUG = True
# Mix data
_C.DATA.MIX = False
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8
# Max length of input audio
_C.DATA.AUDIO_MAX_LENGTH = 0
# Whether to use pretrained language model
_C.DATA.USE_BERT = True
# Transition bias required for uisrnn
_C.DATA.Trans_BIAS = 0.7756

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type :['sernn','rnn', 'transformer','conv']
_C.MODEL.TYPE = 'sernn'
# Model name, auto-renamed later, keep it as 'ours-2*[conv1d(7,1#4,1),shift][False,True]LN'
# ours-speaker_rnn-speaker_contrast+bce-instance_contrast_2smlp-emotion_contrast
_C.MODEL.NAME = 'PCDS'
# Whether to use mask
_C.MODEL.MASK = False
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 4
# Pretrained Model in ['hubert','wav2vec2', 'wavlm']
_C.MODEL.PRETRAIN = 'wav2vec2'
# Dropout rate
_C.MODEL.DROP_RATE = 0.1
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0
# Whether to use temporal shift
_C.MODEL.USE_SHIFT = False
# kernel size of the first layer of convolution
_C.MODEL.KERNEL_SIZE = 7
# Input auido channel
_C.MODEL.AUDIO_DIM = 768
# Input text channel
_C.MODEL.TEXT_DIM = 300
# Hidden channel
_C.MODEL.HIDDEN_DIM = 256
# Input Sequence Length
_C.MODEL.LENGTH = 374
# Temperature for speaker contrast
_C.MODEL.T_S = 1.0
# Temperature for instance contrast
_C.MODEL.T_I = 1.0
# Temperature for supervised emotion contrast
_C.MODEL.T_E = 0.07
# Size of the contrast queue
_C.MODEL.K = 0
# Coefficient of emotion loss
_C.MODEL.ALPHA = 1.0
# Coefficient of speaker diarization loss
_C.MODEL.BETA = 0.5
# Coefficient of cross modal supervised contrastive loss
_C.MODEL.GAMMA = 0.5
# Coefficient of instance contrastive loss
_C.MODEL.DELTA = 0.5
# path to save model
_C.MODEL.SAVE_PATH = './pth'
# whether to save model
_C.MODEL.SAVE = True

# Swin Transformer parameters
_C.MODEL.Trans = CN()
_C.MODEL.Trans.PATCH_SIZE = 4
_C.MODEL.Trans.POSITION = 'relative_key_query'
_C.MODEL.Trans.IN_CHANS = 1
_C.MODEL.Trans.EMBED_DIM = 64
_C.MODEL.Trans.DEPTHS = [1, 2, 1]
_C.MODEL.Trans.NUM_HEADS = [2, 4, 8]
_C.MODEL.Trans.SPLIT_SIZE = [(1, 1), (2, 2), (4, 4)]
_C.MODEL.Trans.MLP_RATIO = 4
_C.MODEL.Trans.ATTN_DROP = 0.
_C.MODEL.Trans.QKV_BIAS = True
_C.MODEL.Trans.PATCH_NORM = True
_C.MODEL.Trans.NUM_TOKENS = 0
_C.MODEL.Trans.USE_LAYER_SCALE = False

# Temporal Shift parameters
_C.MODEL.SHIFT = CN()
_C.MODEL.SHIFT.STRIDE = 1
_C.MODEL.SHIFT.N_DIV = 4
_C.MODEL.SHIFT.BIDIRECTIONAL = False
_C.MODEL.SHIFT.PADDING = 'zero'

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 100
_C.TRAIN.WARMUP_EPOCHS = 5
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 0
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False
# Epochs for teaching
_C.TRAIN.EPOCHS_TEACH = 0

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 10
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
# whether to finetune
_C.TRAIN.FINETUNE = False
# whether to feature extraction
_C.TRAIN.FEATUREX = True

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 42
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0
# fold validation
_C.NUM_FOLD = 5


def ConfigDataset(config):
    config.defrost()
    config.LOGPATH = dataset[config.DATA.DATASET]['log_path']
    config.DATA.DATA_PATH = dataset[config.DATA.DATASET]['data_path']
    config.DATA.AUDIO_MAX_LENGTH = dataset[config.DATA.DATASET]['audio_max_length']
    config.MODEL.NUM_CLASSES = dataset[config.DATA.DATASET]['num_classes']
    config.freeze()


def ConfigPretrain(config):
    config.defrost()
    assert not (config.TRAIN.FINETUNE & config.TRAIN.FEATUREX), "More than 2 modes are adopted!!"
    if config.MODEL.PRETRAIN == 'wav2vec2':
        config.TRAIN.OPTIMIZER.NAME = 'adam'
        config.TRAIN.LR_SCHEDULER.NAME = 'linear'
        config.TRAIN.BASE_LR = 5e-4
    if config.TRAIN.FINETUNE:
        config.TRAIN.OPTIMIZER.NAME = 'adam'
        config.TRAIN.LR_SCHEDULER.NAME = 'lambda'
        # config.TRAIN.BASE_LR = 5e-4
    elif config.TRAIN.FEATUREX:
        config.TRAIN.OPTIMIZER.NAME = 'adam'
        config.TRAIN.LR_SCHEDULER.NAME = 'linear'
        config.TRAIN.BASE_LR = 5e-4
    config.freeze()


def Update(config, args):
    config.defrost()

    if args.dataset:
        config.DATA.DATASET = args.dataset.upper()

    if config.DATA.USE_BERT:
        cfg_file = os.path.join(config.CFGFILE, '{}_bert.yaml'.format(config.DATA.DATASET))
    else:
        cfg_file = os.path.join(config.CFGFILE, '{}_base.yaml'.format(config.DATA.DATASET))
    config.merge_from_file(cfg_file)

    if args.batchsize:
        config.DATA.BATCH_SIZE = args.batchsize
    if args.logpath:
        config.LOGPATH = args.logpath
    if args.logname:
        config.MODEL.NAME = args.logname
    if args.k:
        config.MODEL.K = args.k
    if args.ts:
        config.MODEL.T_S = args.ts
    if args.ti:
        config.MODEL.T_I = args.ti
    if args.te:
        config.MODEL.T_E = args.te
    if args.kernel:
        config.MODEL.KERNEL_SIZE = args.kernel
    if args.shift:
        config.MODEL.USE_SHIFT = True
    if args.stride:
        config.MODEL.SHIFT.STRIDE = args.stride
    if args.ndiv:
        config.MODEL.SHIFT.N_DIV = args.ndiv
    if args.bidirectional:
        config.MODEL.SHIFT.BIDIRECTIONAL = args.bidirectional
    if args.gpu:
        config.LOCAL_RANK = int(args.gpu)
    if args.seed:
        config.SEED = args.seed

    config.freeze()


def Rename(config):
    config.defrost()
    if config.MODEL.NAME == '':
        config.MODEL.NAME = config.MODEL.TYPE
    if config.MODEL.K:
        config.MODEL.NAME = config.MODEL.NAME + '-k' + str(config.MODEL.K)
    config.MODEL.NAME = config.MODEL.NAME + '-t[{},{},{}]'.format(str(config.MODEL.T_S), str(config.MODEL.T_I),
                                                                  str(config.MODEL.T_E))

    if config.TRAIN.FINETUNE:
        config.MODEL.NAME = config.MODEL.NAME + '_finetune' + config.MODEL.PRETRAIN
    elif config.TRAIN.FEATUREX:
        config.MODEL.NAME = config.MODEL.NAME + '_featurex' + config.MODEL.PRETRAIN
    # if config.MODEL.USE_SHIFT:
    #     config.MODEL.NAME = config.MODEL.NAME + '+shift' + str(config.MODEL.SHIFT.N_DIV)
    #     if config.MODEL.SHIFT.BIDIRECTIONAL:
    #         config.MODEL.NAME = config.MODEL.NAME + 'b'
    #     config.MODEL.NAME = config.MODEL.NAME + 'stride' + str(config.MODEL.SHIFT.STRIDE)

    # if config.MODEL.TYPE == 'transformer':
    #     config.MODEL.NAME = config.MODEL.NAME + '-' + config.MODEL.Trans.POSITION

    config.MODEL.NAME = config.MODEL.NAME + '-' + config.DATA.DATA_PATH.split('/')[-1].split('.pkl')[0]

    # config.MODEL.NAME = str(config.SEED) + config.MODEL.NAME
    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    Update(config, args)
    # ConfigDataset(config)
    # ConfigPretrain(config)
    Rename(config)
    return config
