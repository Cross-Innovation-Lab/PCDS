import torch
from torch.utils.data import DataLoader
import pickle
import numpy as np
from dataset import dataset
from config import get_config
import argparse
from transformers import AutoModel, AutoTokenizer
from models.uisrnn import UISRNN
from models.eend import EncoderDecoderAttractor


def parse_option():
    parser = argparse.ArgumentParser('SWin1D', add_help=False)

    # # easy config modification
    # parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    # parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--batchsize',
                        type=int,
                        help="batch size for single GPU")
    parser.add_argument('--logpath', type=str, help='directory path to log')
    parser.add_argument('--logname', type=str, help='name of log')
    parser.add_argument('--dataset',
                        type=str,
                        help='dataset for training/testing')
    parser.add_argument('--shift',
                        action='store_true',
                        help='whether to use temporal shift')
    parser.add_argument('--stride', type=int, help='temporal shift stride')
    parser.add_argument('--ndiv', type=int, help='temporal shift portion')
    parser.add_argument('--k', type=int, help='queue size')
    parser.add_argument('--ts',
                        type=float,
                        help='temperature of speaker contrast')
    parser.add_argument('--ti',
                        type=float,
                        help='temperature of instance contrast')
    parser.add_argument('--te',
                        type=float,
                        help='temperature of emotion contrast')
    parser.add_argument('--bidirectional',
                        action='store_true',
                        help='temporal shift direction')
    parser.add_argument(
        '--kernel',
        type=int,
        help=
        'kernel size of first convolution layer (only work for model type CNN)'
    )
    parser.add_argument('--gpu', type=str, help='gpu rank to use')
    parser.add_argument('--seed', type=int, help='seed')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return config


def build_loader(config):
    dataset_train, dataset_test = build_dataset(config)
    dataloader_train = DataLoader(dataset_train,
                                  batch_size=config.DATA.BATCH_SIZE,
                                  collate_fn=dataset_train.collate_fn)
    dataloader_test = DataLoader(dataset_test,
                                 batch_size=config.DATA.BATCH_SIZE,
                                 collate_fn=dataset_test.collate_fn)
    return dataloader_train, dataloader_test


def build_dataset(config):
    with open(config.DATA.DATA_PATH, 'rb') as f:
        DataMap = pickle.load(f)
    dataset_train = dataset.ERCDATASET(DataMap['train'], config.DATA.USE_BERT,
                                       config.DATA.AUDIO_MAX_LENGTH)
    dataset_test = dataset.ERCDATASET(DataMap['test'], config.DATA.USE_BERT,
                                      config.DATA.AUDIO_MAX_LENGTH)
    return dataset_train, dataset_test


if __name__ == '__main__':
    # model = UISRNN(768)
    # config = parse_option()
    # dataloader_train, dataloader_test = build_loader(config)
    # for idx, (input_audios, input_texts, audio_mask, text_mask, emotions, speakers) in enumerate(dataloader_train):
    #     print(input_audios.shape)
    #     print(input_texts.shape)
    #     print(input_texts)
    #     print(audio_mask.shape)
    #     print(text_mask.shape)
    #     print(text_mask)
    #     print(emotions.shape)
    #     print(speakers.shape)
    #     break

    # pretrain = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    # x = torch.tensor([0.1,0.2,0.4,0.8,0.7,0.5])
    # y = torch.tensor([0,1,1,0,1,0]).float()
    # criterion = torch.nn.BCELoss()
    # print(criterion(x,y))
    # print(criterion(x.view(2,3),y.view(2,3)))
    # print(criterion(x.view(3,2),y.view(3,2)))
    # y = torch.randn((3,10,12))
    # z = torch.nn.utils.rnn.pad_sequence([x,y],batch_first=True)

    # with open('/home/wanghanyang/ssy/MDSER/dataset/IEMOCAP/iemocap_feature.pkl','rb') as f:
    #     DataMap = pickle.load(f)
    # transit_num = 0
    # total_num  = 0
    # for data in DataMap['train']:
    #     for entry in range(len(data['speakers']) - 1):
    #             transit_num += (data['speakers'][entry] != data['speakers'][entry + 1])
    #             total_num += 1
    #     num = len(data['speakers'])
    # print(transit_num / total_num)

    # print(second_num,max_num)
    # tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    # model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    # print(model)
    # intext = 'He is you.'
    # tokens = tokenizer(intext,return_tensors="pt",return_attention_mask=True)
    # print(tokens)
    # res = model(tokens.input_ids)
    # print(res.last_hidden_state.shape)
    # print(res.pooler_output.shape)

    # model = UISRNN(768).cuda(0)
    # x = torch.randn(8, 768).detach().numpy()
    # print(model.predict_single(x))

    # input = torch.randn(5,5).softmax(dim=-1)
    # x = torch.arange(5)
    # y = torch.nn.functional.one_hot(x).float()
    # criterion = torch.nn.CrossEntropyLoss()
    # print(criterion(input,x))
    # print(criterion(input,y))

    # model = UISRNN(768).cuda()
    # x = torch.randn(2,4,768).cuda()
    # mask = torch.ones(2,4).bool().cuda()
    # mask[1][3] = False
    # mask[1][2] = False
    # mask[1][1] = False
    # label = torch.tensor([[2,3,3,4],[1,2,3,0]]).cuda()
    # print(model.predict(x, mask))
    x = torch.randn(2,7)
    y = torch.randn(2,7)
    print(torch.stack((x,y),dim=-1).shape)


