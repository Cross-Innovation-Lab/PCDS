import csv
from email.mime import audio
from tkinter import dialog
import torchaudio
import os
import numpy as np
import pandas as pd
import re
import pickle as pkl
from transformers import Wav2Vec2Config, Wav2Vec2Model, AutoModel, AutoTokenizer
import spacy

sr = 16000

config = Wav2Vec2Config()
model = Wav2Vec2Model(config)
GLOVE_DIR = '../glove.840B.300d.txt'


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


def Raw2Df(indir):
    start_times_final, end_times_final, wav_file_names_final, emotions_final, speakers_final, text_final = [], [], [], [], [], []
    info_line = re.compile(r'\[.+\]\n', re.IGNORECASE)
    text_line = re.compile(r'.*\[(.*)-(.*)\]: (.*)\n')
    for sess in range(1, 6):
        emo_evaluation_dir = '{}Session{}/dialog/EmoEvaluation/'.format(indir, sess)
        transcription_dir = '{}Session{}/dialog/transcriptions/'.format(indir, sess)
        emo_sentences_dir = '{}Session{}/sentences/wav/'.format(indir, sess)
        evaluation_files = [l for l in os.listdir(emo_evaluation_dir) if 'Ses' in l]
        for file in evaluation_files:
            start_times, end_times, wav_file_names, emotions, speakers = [], [], [], [], []
            with open(emo_evaluation_dir + file) as f:
                audio_content = f.read()
            info_lines = re.findall(info_line, audio_content)
            emo_wav_dir = emo_sentences_dir + file.split('.')[0] + '/'
            for line in info_lines[1:]:  # the first line is a header
                start_end_time, wav_file_name, emotion, val_act_dom = line.strip().split('\t')
                gender = wav_file_name.split('_')[-1][0]
                speaker = file.split('.')[0] + '_' + gender
                wav_file_name = emo_wav_dir + wav_file_name + '.wav'
                start_time, end_time = start_end_time[1:-1].split('-')
                start_time, end_time = float(start_time), float(end_time)
                start_times.append(start_time)
                end_times.append(end_time)
                wav_file_names.append(wav_file_name)
                emotions.append(emotion)
                speakers.append(speaker)
            with open(transcription_dir + file) as f:
                text_content = f.read()
            text_lines = re.findall(text_line, text_content)
            start2text = {}
            end2text = {}
            for start, end, text in text_lines:
                start2text[float(start)] = text
                end2text[float(end)] = text

            # assert len(text_lines) == len(start_times), f"dismatch length between the number of audio({len(start_times)}) and text({len(text_lines)}) for {file}"
            idx_ranked = np.argsort(start_times)
            sorted_start_times = [start_times[i] for i in idx_ranked]
            sorted_end_times = [end_times[i] for i in idx_ranked]
            start_times_final.extend(sorted_start_times)
            end_times_final.extend(sorted_end_times)
            wav_file_names_final.extend([wav_file_names[i] for i in idx_ranked])
            emotions_final.extend([emotions[i] for i in idx_ranked])
            speakers_final.extend([speakers[i] for i in idx_ranked])
            text_final.extend([
                end2text[ed] if ed in end2text.keys() else start2text[st]
                for st, ed in zip(sorted_start_times, sorted_end_times)
            ])

    df_iemocap = pd.DataFrame(columns=['start_time', 'end_time', 'wav_file', 'text', 'emotion', 'speaker'])

    df_iemocap['start_time'] = start_times_final
    df_iemocap['end_time'] = end_times_final
    df_iemocap['wav_file'] = wav_file_names_final
    df_iemocap['text'] = text_final
    df_iemocap['emotion'] = emotions_final
    df_iemocap['speaker'] = speakers_final
    return df_iemocap


def Df2Csv(df_iemocap, csvdir):
    df = df_iemocap.copy()
    emotion_id = {'hap': 0, 'sad': 1, 'neu': 2, 'ang': 3, 'exc': 4, 'fru': 5}
    df = df[(df.emotion == 'ang') | (df.emotion == 'sad') | (df.emotion == 'exc') | (df.emotion == 'hap') |
            (df.emotion == 'neu') | (df.emotion == 'fru')]
    df['emotion'] = df['emotion'].map(emotion_id)
    # df['gender'] = df['gender'].map(gender_id)
    df.to_csv(csvdir, index=False, encoding="utf_8_sig")


def process_audio(link):
    wav, sample_rate = torchaudio.load(link)
    wav = wav.squeeze(0).numpy()
    length = wav.shape[0]
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


def Csv2Pickle(csvdir, pikdir):
    SessionMap = {}
    df = pd.read_csv(csvdir, usecols=['wav_file', 'text', 'emotion', 'speaker'])
    glove_embeddings = get_glove_emb()
    nlp = spacy.load('en_core_web_sm')

    for i in range(1, 6):
        sess = 'Session{}'.format(i)
        df_temp = df[df['wav_file'].str.contains(sess)]
        L = df_temp.values.T.tolist()
        audio_links = L[0]
        texts = L[1]
        labels_emotion = L[2]
        labels_speaker = L[3]
        DialogMap = {}
        for link, text, emotion, speaker in zip(audio_links, texts, labels_emotion, labels_speaker):
            wav_vec, audio_length = process_audio(link)
            text_vec, text_length = process_text(text, nlp, glove_embeddings)
            name = link.split('/')[-2]
            if name not in DialogMap.keys():
                DialogMap[name] = {}
                DialogMap[name]['audio'] = [wav_vec]
                DialogMap[name]['text'] = [text_vec]
                DialogMap[name]['emotions'] = [emotion]
                DialogMap[name]['speakers'] = [speaker]
                DialogMap[name]['text_lengths'] = [text_length]
                DialogMap[name]['audio_lengths'] = [audio_length]
            else:
                DialogMap[name]['audio'].append(wav_vec)
                DialogMap[name]['text'].append(text_vec)
                DialogMap[name]['emotions'].append(emotion)
                DialogMap[name]['speakers'].append(speaker)
                DialogMap[name]['text_lengths'].append(text_length)
                DialogMap[name]['audio_lengths'].append(audio_length)
        Dialogs = [v for k, v in DialogMap.items()]
        SessionMap[sess] = Dialogs
        print('Session{} Completed!'.format(i))
    DataMap = {}
    DataMap['test'] = SessionMap['Session{}'.format(5)]  # data for test
    DataMap['train'] = []
    for i in range(1, 6):
        sess = 'Session{}'.format(i)
        if i != 5:
            DataMap['train'].extend(SessionMap[sess])
    with open(pikdir, 'wb') as f:
        pkl.dump(DataMap, f)

def Csv2Pickle4Bert(csvdir, pikdir):
    SessionMap = {}
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    df = pd.read_csv(csvdir, usecols=['wav_file', 'text', 'emotion', 'speaker'])

    for i in range(1, 6):
        sess = 'Session{}'.format(i)
        df_temp = df[df['wav_file'].str.contains(sess)]
        L = df_temp.values.T.tolist()
        audio_links = L[0]
        texts = L[1]
        labels_emotion = L[2]
        labels_speaker = L[3]
        DialogMap = {}
        for link, text, emotion, speaker in zip(audio_links, texts, labels_emotion, labels_speaker):
            wav_vec, audio_length = process_audio(link)
            text = re.sub('\s+', ' ', text.strip())
            text_output = tokenizer(text, max_length=128, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')
            name = link.split('/')[-2]
            if name not in DialogMap.keys():
                DialogMap[name] = {}
                DialogMap[name]['audio'] = [wav_vec]
                DialogMap[name]['text'] = [text_output.input_ids[0]]
                DialogMap[name]['emotions'] = [emotion]
                DialogMap[name]['speakers'] = [speaker]
                DialogMap[name]['text_lengths'] = [text_output.attention_mask[0]]
                DialogMap[name]['audio_lengths'] = [audio_length]
            else:
                DialogMap[name]['audio'].append(wav_vec)
                DialogMap[name]['text'].append(text_output.input_ids[0])
                DialogMap[name]['emotions'].append(emotion)
                DialogMap[name]['speakers'].append(speaker)
                DialogMap[name]['text_lengths'].append(text_output.attention_mask[0])
                DialogMap[name]['audio_lengths'].append(audio_length)
        Dialogs = [v for k, v in DialogMap.items()]
        SessionMap[sess] = Dialogs
        print('Session{} Completed!'.format(i))
    DataMap = {}
    DataMap['test'] = SessionMap['Session{}'.format(5)]  # data for test
    DataMap['train'] = []
    for i in range(1, 6):
        sess = 'Session{}'.format(i)
        if i != 5:
            DataMap['train'].extend(SessionMap[sess])
    with open(pikdir, 'wb') as f:
        pkl.dump(DataMap, f)

if __name__ == '__main__':
    indir = '/home/wanghanyang/Public/DATASET/IEMOCAP/IEMOCAP/'
    csvdir = 'IEMOCAP.csv'
    # pikdir = 'iemocap_feature.pkl'
    # print(f'output pickle: {pikdir}')
    # df_iemocap = Raw2Df(indir)
    # Df2Csv(df_iemocap, csvdir)
    # Csv2Pickle(csvdir, pikdir)

    pikdir = './iemocap_bert_feature.pkl'
    print(f'output pickle: {pikdir}')
    Csv2Pickle4Bert(csvdir, pikdir)
