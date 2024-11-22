from unicodedata import bidirectional
from .rnn import RNN
from .linear import Linear
from .sernn import SERNN


def build_model(config):
    model_type = config.MODEL.TYPE
    # finetune = config.TRAIN.FINETUNE
    # featurex = config.TRAIN.FEATUREX
    # if model_type == 'transformer':
    #     model = Transformer(dim=config.DATA.DIM,
    #                         length=config.DATA.LENGTH,
    #                         num_classes=config.MODEL.NUM_CLASSES,
    #                         shift=config.MODEL.USE_SHIFT,
    #                         stride=config.MODEL.SHIFT.STRIDE,
    #                         n_div=config.MODEL.SHIFT.N_DIV,
    #                         bidirectional=config.MODEL.SHIFT.BIDIRECTIONAL,
    #                         mlp_ratio=config.MODEL.Trans.MLP_RATIO,
    #                         mask=config.MODEL.MASK,
    #                         drop=config.MODEL.DROP_RATE,
    #                         position_embedding_type=config.MODEL.Trans.POSITION,
    #                         use_layer_scale=config.MODEL.Trans.USE_LAYER_SCALE)
    # elif model_type == 'rnn':
    #     model = RNN(dim=config.DATA.DIM,
    #                 length=config.DATA.LENGTH,
    #                 num_classes=config.MODEL.NUM_CLASSES,
    #                 shift=config.MODEL.USE_SHIFT,
    #                 stride=config.MODEL.SHIFT.STRIDE,
    #                 n_div=config.MODEL.SHIFT.N_DIV,
    #                 bidirectional=config.MODEL.SHIFT.BIDIRECTIONAL,
    #                 mask=config.MODEL.MASK,
    #                 drop=config.MODEL.DROP_RATE)
    # elif model_type == 'linear':
    #     model = Linear(dim=config.DATA.DIM, num_classes=config.MODEL.NUM_CLASSES)
    # else:
    #     raise NotImplementedError(f"Unkown model: {model_type}")

    # if finetune:
    #     model = Pretrain_Model(model, config.DATA.DIM, config.MODEL.PRETRAIN, True, config.MODEL.DROP_RATE)
    # elif featurex:
    #     model = Pretrain_Model(model, config.DATA.DIM, config.MODEL.PRETRAIN, False, config.MODEL.DROP_RATE)
    if model_type == 'sernn':
        model = SERNN(audio_dim=config.MODEL.AUDIO_DIM,
                      text_dim=config.MODEL.TEXT_DIM,
                      hidden_dim=config.MODEL.HIDDEN_DIM,
                      alpha=config.MODEL.ALPHA,
                      beta=config.MODEL.BETA,
                      gamma=config.MODEL.GAMMA,
                      delta = config.MODEL.DELTA,
                      T_s=config.MODEL.T_S,
                      T_i=config.MODEL.T_I,
                      T_e=config.MODEL.T_E,
                      K=config.MODEL.K,
                      num_classes=config.MODEL.NUM_CLASSES,
                      use_bert=config.DATA.USE_BERT,
                      pretrain=config.MODEL.PRETRAIN,
                      dataset=config.DATA.DATASET,
                      finetune=config.TRAIN.FINETUNE,
                      drop=config.MODEL.DROP_RATE)
    return model