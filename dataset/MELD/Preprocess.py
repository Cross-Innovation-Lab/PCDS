import csv
from email.mime import audio
from tkinter import dialog
import torch
import torchaudio
import os
import numpy as np
import pandas as pd
import re
import pickle as pkl
from transformers import Wav2Vec2Config, Wav2Vec2Model, AutoModel, AutoTokenizer
import spacy

sample_rate = 48000
target_sample_rate = 16000

config = Wav2Vec2Config()
model = Wav2Vec2Model(config)


GLOVE_DIR = '../glove.840B.300d.txt'
L = []


def get_feat_extract_output_lengths(input_length):
    """
        Computes the output length of the convolutional layers
        """

    def _conv_out_length(input_length, kernel_size, stride):
        # 1D convolutional layer output length formula taken
        # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        return (input_length - kernel_size) // stride + 1

    for kernel_size, stride in zip(model.config.conv_kernel, model.config.conv_stride):
        input_length = _conv_out_length(input_length, kernel_size, stride)
    return input_length


def process_audio(link, transform):
    wav, sample_rate = torchaudio.load(link)
    wav = torch.mean(transform(wav),dim=0,keepdim=True)
    length = wav.shape[1]
    wav = wav.squeeze(0).numpy()
    output_length = get_feat_extract_output_lengths(length)
    return wav, output_length


def process_text(text, nlp, glove_embeddings):
    text = re.sub('\s+', ' ', text.strip())
    vec = np.array([
        glove_embeddings[token.text.lower()]
        if token.text.lower() in glove_embeddings.keys() else glove_embeddings['<unk>'] for token in nlp(text)
    ])
    return vec, len(vec)


def get_glove_emb():
    glove_embeddings = {}
    with open(GLOVE_DIR, 'rb') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            glove_embeddings[word.decode().lower()] = coefs
        glove_embeddings['<unk>'] = np.random.uniform(-0.25, 0.25, 300)
        return glove_embeddings


def Csv2Pickle(datadir, pikdir):
    emotion_id = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}
    glove_embeddings = get_glove_emb()
    nlp = spacy.load('en_core_web_sm')
    transform = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
    DataMap = {}

    for key, value in datadir.items():
        csvdir = value[0]
        wavdir = value[1]
        DialogMap = {}
        df = pd.read_csv(csvdir, usecols=['Utterance', 'Speaker', 'Emotion', 'Dialogue_ID', 'Utterance_ID'])
        texts = df['Utterance'].to_list()
        labels_speaker = df['Speaker'].to_list()
        labels_emotion = df['Emotion'].map(emotion_id).to_list()
        dialogids = df['Dialogue_ID'].to_list()
        utterids = df['Utterance_ID'].to_list()
        for dialogid, utterid, text, emotion, speaker in zip(dialogids, utterids, texts, labels_emotion,
                                                             labels_speaker):
            link = os.path.join(wavdir, 'dia{}_utt{}.wav'.format(dialogid, utterid))
            if not os.path.exists(link):
                continue
            wav_vec, audio_length = process_audio(link, transform)
            text_vec, text_length = process_text(text, nlp, glove_embeddings)
            if dialogid not in DialogMap:
                DialogMap[dialogid] = {}
                DialogMap[dialogid]['audio'] = [wav_vec]
                DialogMap[dialogid]['text'] = [text_vec]
                DialogMap[dialogid]['emotions'] = [emotion]
                DialogMap[dialogid]['speakers'] = [speaker]
                DialogMap[dialogid]['text_lengths'] = [text_length]
                DialogMap[dialogid]['audio_lengths'] = [audio_length]
            else:
                DialogMap[dialogid]['audio'].append(wav_vec)
                DialogMap[dialogid]['text'].append(text_vec)
                DialogMap[dialogid]['emotions'].append(emotion)
                DialogMap[dialogid]['speakers'].append(speaker)
                DialogMap[dialogid]['text_lengths'].append(text_length)
                DialogMap[dialogid]['audio_lengths'].append(audio_length)
        Dialogs = [v for k, v in DialogMap.items()]
        DataMap[key] = Dialogs
        print('{}set Completed!'.format(key))
    with open(pikdir, 'wb') as f:
        pkl.dump(DataMap, f)

def Csv2Pickle4Bert(datadir, pikdir):
    emotion_id = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    transform = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
    DataMap = {}

    for key, value in datadir.items():
        csvdir = value[0]
        wavdir = value[1]
        DialogMap = {}
        df = pd.read_csv(csvdir, usecols=['Utterance', 'Speaker', 'Emotion', 'Dialogue_ID', 'Utterance_ID'])
        texts = df['Utterance'].to_list()
        labels_speaker = df['Speaker'].to_list()
        labels_emotion = df['Emotion'].map(emotion_id).to_list()
        dialogids = df['Dialogue_ID'].to_list()
        utterids = df['Utterance_ID'].to_list()
        for dialogid, utterid, text, emotion, speaker in zip(dialogids, utterids, texts, labels_emotion,
                                                             labels_speaker):
            link = os.path.join(wavdir, 'dia{}_utt{}.wav'.format(dialogid, utterid))
            if not os.path.exists(link):
                continue
            wav_vec, audio_length = process_audio(link, transform)
            text = re.sub('\s+', ' ', text.strip())
            text_output = tokenizer(text, max_length=128, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')
            if dialogid not in DialogMap:
                DialogMap[dialogid] = {}
                DialogMap[dialogid]['audio'] = [wav_vec]
                DialogMap[dialogid]['text'] = [text_output.input_ids[0]]
                DialogMap[dialogid]['emotions'] = [emotion]
                DialogMap[dialogid]['speakers'] = [speaker]
                DialogMap[dialogid]['text_lengths'] = [text_output.attention_mask[0]]
                DialogMap[dialogid]['audio_lengths'] = [audio_length]
            else:
                DialogMap[dialogid]['audio'].append(wav_vec)
                DialogMap[dialogid]['text'].append(text_output.input_ids[0])
                DialogMap[dialogid]['emotions'].append(emotion)
                DialogMap[dialogid]['speakers'].append(speaker)
                DialogMap[dialogid]['text_lengths'].append(text_output.attention_mask[0])
                DialogMap[dialogid]['audio_lengths'].append(audio_length)
        Dialogs = [v for k, v in DialogMap.items()]
        DataMap[key] = Dialogs
        print('{}set Completed!'.format(key))
    with open(pikdir, 'wb') as f:
        pkl.dump(DataMap, f)


if __name__ == '__main__':
    train_csvdir = '/home/wanghanyang/Public/DATASET/MELD/train_sent_emo.csv'
    train_wavdir = '/home/wanghanyang/Public/DATASET/MELD/train_splits_wav'
    dev_csvdir = '/home/wanghanyang/Public/DATASET/MELD/dev_sent_emo.csv'
    dev_wavdir = '/home/wanghanyang/Public/DATASET/MELD/dev_splits_wav'
    test_csvdir = '/home/wanghanyang/Public/DATASET/MELD/test_sent_emo.csv'
    test_wavdir = '/home/wanghanyang/Public/DATASET/MELD/test_splits_wav'
    pikdir = './meld_feature.pkl'
    datadir = {
        'train': (train_csvdir, train_wavdir),
        'dev': (dev_csvdir, dev_wavdir),
        'test': (test_csvdir, test_wavdir)
    }
    # pikdir = './meld_feature.pkl'
    # print(f'output pickle: {pikdir}')
    # Csv2Pickle(datadir, pikdir)

    pikdir = './meld_bert_feature.pkl'
    print(f'output pickle: {pikdir}')
    Csv2Pickle4Bert(datadir, pikdir)

