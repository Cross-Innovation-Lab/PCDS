import torch
import torch.utils.data as data
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Config, Wav2Vec2Model
from torch.nn.utils.rnn import pad_sequence

config = Wav2Vec2Config()
model = Wav2Vec2Model(config)


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


def get_padding_mask(max_len, batch_size, lengths):
    """Generate the padding mask given the padded input and the lengths Tensors.
    Args:
        lengths (Tensor): The lengths Tensor of dimension `[batch,]`.

    Returns:
        (Tensor): The padding mask.
    """
    mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) < lengths[:, None]
    return mask


class ERCDATASET(data.Dataset):
    """Speech dataset."""
    def __init__(self, data, use_bert, audio_max_length):
        self.data = data
        self.audio_max_length = audio_max_length  # 7.5s for iemocap 3.5s for meld
        self.text_max_length = 128
        self.len = len(self.data)
        self.use_bert = use_bert
        self.extractor = Wav2Vec2FeatureExtractor(sampling_rate=16000, return_attention_mask=False, do_normalize=False)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_wav_feature(self, wavs):
        '''
        wavs: list of 1-d numpy array with size (num_frames,)
        '''
        inputs = self.extractor(wavs,
                                max_length=self.audio_max_length,
                                padding='max_length',
                                truncation=True,
                                return_tensors="pt",
                                sampling_rate=16000)
        inputs = inputs.to(self.device)
        return inputs

    def get_text_feature(self, texts):
        '''
        texts: list of 2-d numpy array with size (num_words, dim)
        '''
        inputs = []
        for sentence in texts:
            num_words = sentence.shape[0]
            difference = self.text_max_length - num_words
            if difference > 0:
                sentence = np.pad(sentence, ((0, difference), (0, 0)))
            else:
                sentence = sentence[:self.text_max_length, :]
            inputs.append(sentence)
        inputs = np.stack(inputs)
        return torch.from_numpy(inputs).float().to(self.device)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data[idx]['audio'], self.data[idx]['text'], self.data[idx]['emotions'], self.data[idx][
            'speakers'], self.data[idx]['audio_lengths'], self.data[idx]['text_lengths']

    def collate_fn(self, datas):
        audio_feature = [self.get_wav_feature(data[0]) for data in datas]
        input_audios = pad_sequence([feature['input_values'] for feature in audio_feature],
                                    batch_first=True,
                                    padding_value=0)

        # attention_mask = pad_sequence([feature['attention_mask'] for feature in batchfeature],
        #                               batch_first=True,
        #                               padding_value=0)
        emotions = pad_sequence([torch.tensor(data[2], device=input_audios.device) for data in datas],
                                batch_first=True,
                                padding_value=-100)

        speakers_str = []
        for data in datas:
            speakers_str.extend(data[3])
        global_id = 0
        for batch_index, data in enumerate(datas):
            id_per_dialogue = {}
            for speaker_index, id in enumerate(data[3]):
                if id not in id_per_dialogue.keys():
                    id_per_dialogue[id] = global_id
                    global_id += 1
                datas[batch_index][3][speaker_index] = id_per_dialogue[id]

        speakers = pad_sequence([torch.tensor(data[3], device=input_audios.device) for data in datas],
                                batch_first=True,
                                padding_value=-100)
        audio_max_output_lengths = get_feat_extract_output_lengths(self.audio_max_length)
        audio_mask = pad_sequence([
            get_padding_mask(audio_max_output_lengths, len(data[4]), torch.tensor(data[4], device=input_audios.device))
            for data in datas
        ],
                                  batch_first=True,
                                  padding_value=0)

        if self.use_bert:
            text_feature = [torch.stack(data[1]).to(self.device) for data in datas]
            input_texts = pad_sequence(text_feature, batch_first=True, padding_value=0)
            text_mask = pad_sequence([torch.stack(data[5]).to(self.device) for data in datas],
                                     batch_first=True,
                                     padding_value=0)
        else:
            text_feature = [self.get_text_feature(data[1]) for data in datas]
            input_texts = pad_sequence(text_feature, batch_first=True, padding_value=0)
            text_mask = pad_sequence([
                get_padding_mask(self.text_max_length, len(data[5]), torch.tensor(data[5], device=input_audios.device))
                for data in datas
            ],
                                     batch_first=True,
                                     padding_value=0)

        return input_audios, input_texts, audio_mask, text_mask, emotions, speakers, speakers_str
