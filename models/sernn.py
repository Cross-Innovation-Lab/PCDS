from turtle import forward
import os
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.file_utils import ModelOutput
from transformers import Wav2Vec2Model, Wav2Vec2Config, HubertModel, HubertConfig, WavLMModel, WavLMConfig
from .utils import Specaugment
from dataclasses import dataclass
from typing import Optional, Tuple, List
from .rnn import SpeakerRNN, Clusterer, CClusterer
from .eend import EncoderDecoderAttractor
from .uisrnn import UISRNN
from .pretrain import Speech_Pretrain_Model, Text_Pretrain_Model
from .transformer import Attention
from .loss import SupConLoss


@dataclass
class ClsOutput(ModelOutput):
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    pred_speakers: torch.FloatTensor = None
    context_mask: torch.FloatTensor = None
    cl_loss: Optional[torch.FloatTensor] = None
    gen_loss: Optional[torch.FloatTensor] = None


class Permute(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, self.dims)


class ConvBlock(nn.Module):
    def __init__(self, dim, kernel_size=3) -> None:
        super().__init__()
        depth_conv = nn.Conv1d(dim, dim, kernel_size, 1, kernel_size // 2, groups=dim)
        point_conv1 = nn.Conv1d(dim, dim, 1)
        point_conv2 = nn.Conv1d(dim, dim, 1)
        self.sequential = nn.Sequential(Permute([0, 2, 1]), depth_conv, Permute([0, 2, 1]), nn.LayerNorm(dim),
                                        Permute([0, 2, 1]), point_conv1, nn.GELU(), point_conv2, Permute([0, 2, 1]))

    def forward(self, x):
        '''
        x: batch_size, seq_num, dim
        '''
        out = self.sequential(x)
        return out


class SERNN(nn.Module):
    def __init__(self,
                 audio_dim,
                 text_dim,
                 hidden_dim,
                 alpha,
                 beta,
                 gamma,
                 delta,
                 T_s=1.0,
                 T_i=1.0,
                 T_e=0.07,
                 K=256,
                 num_classes=4,
                 use_bert=False,
                 pretrain='hubert',
                 dataset='IEMOCAP',
                 finetune=False,
                 drop=0.):
        super().__init__()
        self.speech_pretrain_model = Speech_Pretrain_Model(audio_dim, pretrain, finetune)
        self.use_bert = use_bert
        if self.use_bert:
            self.text_pretrain_model = Text_Pretrain_Model(finetune)
        else:
            self.text_project = nn.Linear(text_dim, hidden_dim)
            self.audio_project = nn.Linear(audio_dim, hidden_dim)
        # self.specaugment = Specaugment(audio_dim)

        self.lowernn = LOWERNN(hidden_dim)

        self.audio_context_rnn = nn.LSTM(input_size=hidden_dim,
                                         hidden_size=hidden_dim // 2,
                                         num_layers=1,
                                         batch_first=True,
                                         bidirectional=True,
                                         dropout=drop)

        self.text_context_rnn = nn.LSTM(input_size=hidden_dim,
                                        hidden_size=hidden_dim // 2,
                                        num_layers=1,
                                        batch_first=True,
                                        bidirectional=True,
                                        dropout=drop)

        # supervised clustering
        self.clusterer = CClusterer(hidden_dim, T_s)
        # self.clusterer = EncoderDecoderAttractor(hidden_dim)

        # self.clusterer = None
        # self.deepsup_loss1 = CrossEntropyLoss() # deep supervision
        # self.deepsup_head1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, num_classes))
        # self.condeepsup_loss1 = SupConLoss(temperature=1.0, contrast_mode='all') # contrast deep supervision
        # self.condeepsup_head1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, hidden_dim))
        # self.clusterer = UISRNN(hidden_dim, 0.7756)  # meld-0.7756, iemocap-0.6751

        self.text_speaker_rnn = SpeakerRNN(hidden_dim)
        self.audio_speaker_rnn = SpeakerRNN(hidden_dim)

        # intermediate
        # sup_dim = hidden_dim
        sup_dim = 2 * hidden_dim
        self.text_linear = nn.Sequential(nn.Linear(sup_dim, sup_dim))
        self.auido_linear = nn.Sequential(nn.Linear(sup_dim, sup_dim))

        # self.ic_head_text = nn.Sequential(nn.Linear(sup_dim, sup_dim // 2), nn.GELU(), nn.Linear(sup_dim // 2, sup_dim))
        # self.ic_head_audio = nn.Sequential(nn.Linear(sup_dim, sup_dim // 2), nn.GELU(),
        #                                    nn.Linear(sup_dim // 2, sup_dim))

        self.criterion_ic = SupConLoss(temperature=T_i, contrast_mode='all') # unsupervised contrast
        # self.criterion_ic = SupConLoss(temperature=T_i) # supervised contrast

        # self.criterion_ic = CrossEntropyLoss() # deep supervision
        self.ic_head_text = nn.Sequential(nn.Linear(sup_dim, sup_dim // 2), nn.GELU(), nn.Linear(sup_dim // 2, num_classes))
        self.ic_head_audio = nn.Sequential(nn.Linear(sup_dim, sup_dim // 2), nn.GELU(),
                                            nn.Linear(sup_dim // 2, num_classes))
        
        # self.intermidiate_text = nn.Linear(sup_dim, sup_dim)
        # self.intermidiate_audio = nn.Linear(sup_dim, sup_dim)
        if dataset == 'IEMOCAP':
            self.intermidiate = nn.Linear(sup_dim, sup_dim)
        elif dataset == 'MELD':
            self.intermidiate = nn.Sequential(nn.Linear(sup_dim, sup_dim), nn.GELU(), nn.Linear(sup_dim, sup_dim))

        # supervised contrast
        # self.criterion_sc = SupConLoss(temperature=T, K=K)
        self.criterion_sc = SupConLoss(temperature=T_e)
        # self.contrast_head = nn.Sequential(nn.Linear(sup_dim,sup_dim))

        # concat
        # post_dim = 2 * hidden_dim
        post_dim = 2 * sup_dim

        # concat + attention
        # self.utt_attention = Attention(sup_dim,4)

        self.norm = nn.LayerNorm(post_dim)
        self.head = nn.Linear(post_dim, num_classes)
        self.criterion_ce = CrossEntropyLoss()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def forward(self, input_audios, input_texts, audio_mask, text_mask, emotions, speakers):
        '''
        input_audios : batch_size, seq_num, seq_len
        input_texts : batch_size, seq_num, seq_len, dim / batch_size, seq_num, seq_len
        audio_mask : batch_size, seq_num, seq_len
        text_mask : batch_size, seq_num, seq_len
        emotions : batch_size, seq_num
        '''
        audio_context_mask = torch.sum(audio_mask, dim=-1).gt(0)  # [batch_size, seq_num]
        audio_seqlens = torch.sum(audio_context_mask, dim=-1)  # [batch_size]
        text_context_mask = torch.sum(text_mask, dim=-1).gt(0)  # [batch_size, seq_num]
        text_seqlens = torch.sum(text_context_mask, dim=-1)  # [batch_size]
        x_audio, x_speaker = self.speech_pretrain_model(input_audios[audio_context_mask, :])

        x_audio_aug, x_speaker_aug = self.speech_pretrain_model(input_audios[audio_context_mask, :].clone())

        # x_audio = self.specaugment(x_audio)
        # x_audio_aug = self.specaugment(x_audio_aug)
        # x_speaker = self.specaugment(x_speaker)
        # x_speaker_aug = self.specaugment(x_speaker_aug)

        # assert torch.equal(audio_context_mask.data, text_context_mask.data), "unmatched mask!"

        if self.use_bert:
            x_text = self.text_pretrain_model(input_texts[text_context_mask], text_mask[text_context_mask])
            x_text_aug = self.text_pretrain_model(input_texts[text_context_mask], text_mask[text_context_mask])
        else:
            x_text = self.text_project(input_texts)[text_context_mask]
            x_audio = self.audio_project(x_audio)
            x_speaker = self.audio_project(x_speaker)
            x_text_aug = self.text_project(input_texts.clone())[text_context_mask]
            x_audio_aug = self.audio_project(x_audio_aug)
            x_speaker_aug = self.audio_project(x_speaker_aug)

        # Frame/Word level encoding and fusion, return [batch_size, seq_num, dim]
        x_text, x_text_fuse, x_audio, x_speaker = self.lowernn(x_text, x_audio, x_speaker, text_mask, text_context_mask,
                                                               text_seqlens, audio_mask, audio_context_mask,
                                                               audio_seqlens, self.use_bert)
        x_text_aug, x_text_fuse_aug, x_audio_aug, x_speaker_aug = self.lowernn(x_text_aug, x_audio_aug, x_speaker_aug,
                                                                               text_mask, text_context_mask,
                                                                               text_seqlens, audio_mask,
                                                                               audio_context_mask, audio_seqlens,
                                                                               self.use_bert)

        # Speaker information (Clustering)
        if isinstance(self.clusterer, CClusterer):
            pred_speakers, loss_affinity = self.clusterer(x_speaker, audio_context_mask, speakers, x_speaker_aug)
        elif isinstance(self.clusterer, EncoderDecoderAttractor):
            pred_speakers, loss_affinity = self.clusterer(x_speaker, audio_context_mask, speakers)
        elif isinstance(self.clusterer, UISRNN):
            if self.training:
                pred_speakers, loss_affinity = self.clusterer(x_speaker, audio_context_mask, speakers)
            else:
                pred_speakers, loss_affinity = self.clusterer.predict(x_speaker, audio_context_mask)
        else:
            pred_speakers = speakers
            # loss_affinity = 0.5 * self.deepsup_loss1(self.deepsup_head1(
            #     x_speaker[audio_context_mask, :]), emotions[audio_context_mask]) + 0.5 * self.deepsup_loss1(
            #         self.deepsup_head1(x_text[text_context_mask, :]), emotions[text_context_mask])
            # loss_affinity = 0.5 * self.condeepsup_loss1(
            #     self.condeepsup_head1(
            #         torch.stack((x_speaker[audio_context_mask, :], x_speaker_aug[audio_context_mask, :]), dim=1)),
            #     emotions[audio_context_mask]) + 0.5 * self.condeepsup_loss1(
            #         self.condeepsup_head1(
            #             torch.stack((x_text_fuse[text_context_mask, :], x_text_fuse_aug[text_context_mask, :]), dim=1)),
            #         emotions[text_context_mask])
            loss_affinity = 0.5 * self.condeepsup_loss1(
                self.condeepsup_head1(
                    torch.stack((x_speaker[audio_context_mask, :], x_speaker_aug[audio_context_mask, :]), dim=1))) + 0.5 * self.condeepsup_loss1(
                    self.condeepsup_head1(
                        torch.stack((x_text_fuse[text_context_mask, :], x_text_fuse_aug[text_context_mask, :]), dim=1)))
        # pred_speakers = speakers
        text_speaker_output = self.text_speaker_rnn(x_text_fuse, text_context_mask, pred_speakers)
        audio_speaker_output = self.audio_speaker_rnn(x_speaker, audio_context_mask, pred_speakers)
        if self.training:
            text_speaker_output_aug = self.text_speaker_rnn(x_text_fuse_aug, text_context_mask, pred_speakers)
            audio_speaker_output_aug = self.audio_speaker_rnn(x_speaker_aug, audio_context_mask, pred_speakers)

        # Contextual information
        audio_output = encode_with_rnn(self.audio_context_rnn, x_audio, audio_seqlens)
        text_output = encode_with_rnn(self.text_context_rnn, x_text, text_seqlens)
        if self.training:
            audio_output_aug = encode_with_rnn(self.audio_context_rnn, x_audio_aug, audio_seqlens)
            text_output_aug = encode_with_rnn(self.text_context_rnn, x_text_aug, text_seqlens)

        text_infeat = self.text_linear(torch.cat((text_output, text_speaker_output), dim=-1))
        audio_infeat = self.auido_linear(torch.cat((audio_output, audio_speaker_output), dim=-1))
        if self.training:
            text_infeat_aug = self.text_linear(torch.cat((text_output_aug, text_speaker_output_aug), dim=-1))
            audio_infeat_aug = self.auido_linear(torch.cat((audio_output_aug, audio_speaker_output_aug), dim=-1))

        # text_infeat = text_output
        # audio_infeat = audio_output
        # text_infeat_aug = text_output_aug
        # audio_infeat_aug = audio_output_aug

        # Instance Contrast
        text_infeat_norm = F.normalize(self.ic_head_text(text_infeat), dim=-1)
        audio_infeat_norm = F.normalize(self.ic_head_audio(audio_infeat), dim=-1)
        if self.training:
            text_infeat_norm_aug = F.normalize(self.ic_head_text(text_infeat_aug), dim=-1)
            audio_infeat_norm_aug = F.normalize(self.ic_head_audio(audio_infeat_aug), dim=-1)
            
        # unsupervised contrast
            loss_incontrast_text = self.criterion_ic(
                torch.stack((text_infeat_norm[text_context_mask, :], text_infeat_norm_aug[text_context_mask, :]), dim=1))
            loss_incontrast_audio = self.criterion_ic(
                torch.stack((audio_infeat_norm[audio_context_mask, :], audio_infeat_norm_aug[audio_context_mask, :]),
                            dim=1))

        # supervised contrast
        # loss_incontrast_text = self.criterion_ic(
        #     torch.stack((text_infeat_norm[text_context_mask, :], text_infeat_norm_aug[text_context_mask, :]), dim=1),
        #     emotions[audio_context_mask])
        # loss_incontrast_audio = self.criterion_ic(
        #     torch.stack((audio_infeat_norm[audio_context_mask, :], audio_infeat_norm_aug[audio_context_mask, :]),
        #                 dim=1), emotions[text_context_mask])
        

        # Deep Supervised
        # loss_incontrast_text = self.criterion_ic(self.ic_head_text(text_infeat)[text_context_mask, :], emotions[text_context_mask])
        # loss_incontrast_audio = self.criterion_ic(self.ic_head_audio(audio_infeat)[audio_context_mask, :], emotions[audio_context_mask])


            loss_incontrast = 0.5 * loss_incontrast_text + 0.5 * loss_incontrast_audio
        else:
            loss_incontrast = 0

        text_feat = self.intermidiate(text_infeat)
        audio_feat = self.intermidiate(audio_infeat)
        # text_feat = self.intermidiate_text(text_infeat)
        # audio_feat = self.intermidiate_audio(audio_infeat)

        # Emotion Supervised Contrast

        text_supfeat = F.normalize(text_feat, dim=-1)
        audio_supfeat = F.normalize(audio_feat, dim=-1)
        loss_contrast_text = self.criterion_sc(
            torch.stack((text_supfeat[text_context_mask, :], audio_supfeat[audio_context_mask, :]), dim=1),
            emotions[audio_context_mask])
        loss_contrast_audio = self.criterion_sc(
            torch.stack((audio_supfeat[audio_context_mask, :], text_supfeat[text_context_mask, :]), dim=1),
            emotions[audio_context_mask])
        loss_contrast = 0.5 * loss_contrast_text + 0.5 * loss_contrast_audio

        # Emotion classification

        logits = self.head(self.norm(torch.cat((text_feat, audio_feat), dim=-1)))
        loss_emotion = self.criterion_ce(logits[audio_context_mask, :], emotions[audio_context_mask])

        return ClsOutput(loss=self.alpha * loss_emotion + self.beta * loss_affinity + self.gamma * loss_contrast +
                         self.delta * loss_incontrast,
                         logits=logits[audio_context_mask, :],
                         pred_speakers=pred_speakers,
                         context_mask=audio_context_mask)
        # return ClsOutput(loss=self.alpha * loss_emotion + self.gamma * loss_contrast + self.delta * loss_incontrast
        #                  ,
        #                  logits=logits[audio_context_mask, :],
        #                  pred_speakers=pred_speakers,
        #                  context_mask=audio_context_mask)


class LOWERNN(nn.Module):
    def __init__(self, hidden_dim, drop=0.):
        super().__init__()
        self.audio_local_rnn = nn.LSTM(input_size=hidden_dim,
                                       hidden_size=hidden_dim // 2,
                                       num_layers=1,
                                       batch_first=True,
                                       bidirectional=True,
                                       dropout=drop)
        self.text_local_rnn = nn.LSTM(input_size=hidden_dim,
                                      hidden_size=hidden_dim // 2,
                                      num_layers=1,
                                      batch_first=True,
                                      bidirectional=True,
                                      dropout=drop)
        self.frame_cross_attention = Attention(hidden_dim, 4)
        self.word_cross_attention = Attention(hidden_dim, 4)

    def forward(self, x_text, x_audio, x_speaker, text_mask, text_context_mask, text_seqlens, audio_mask,
                audio_context_mask, audio_seqlens, use_bert):
        '''
        x_/text/audio/speaker : seq_num, seq_len, dim
        audio_mask : batch_size, seq_num, seq_len
        audio_context_mask : batch_size, seq_num
        audio_seqlens: batch_size
        text_mask : batch_size, seq_num, seq_len
        text_context_mask : batch_size, seq_num
        text_seqlens: batch_size
        '''
        frame_mask = audio_mask[audio_context_mask, :]  # [seq_num, seq_len]
        word_mask = text_mask[text_context_mask, :]  # [seq_num, seq_len]
        # shape of x_audio/speaker/text: [seq_num, seq_len, dim]
        # output of pool_and_pad: [batch_size, seq_num, dim]
        x_audio = encode_with_rnn(self.audio_local_rnn, x_audio, torch.sum(frame_mask, dim=-1))

        x_speaker = encode_with_rnn(self.audio_local_rnn, x_speaker, torch.sum(frame_mask, dim=-1))

        x_text = encode_with_rnn(self.text_local_rnn, x_text, torch.sum(word_mask, dim=-1))

        x_speaker = x_speaker + self.frame_cross_attention(x_speaker, frame_mask, x_text, word_mask)
        x_text_fuse = x_text + self.word_cross_attention(x_text, word_mask, x_speaker, frame_mask)
        # x_text_fuse = x_text
        # x_text_fuse = x_text

        x_audio = pool_and_pad(x_audio, audio_mask, audio_context_mask, audio_seqlens)
        x_speaker = pool_and_pad(x_speaker, audio_mask, audio_context_mask, audio_seqlens)
        # x_audio_fuse = pool_and_pad(x_audio_fuse, audio_mask, audio_context_mask, audio_seqlens)
        x_text = pool_and_pad(x_text, text_mask, text_context_mask, text_seqlens, use_bert)
        x_text_fuse = pool_and_pad(x_text_fuse, text_mask, text_context_mask, text_seqlens, use_bert)
        return x_text, x_text_fuse, x_audio, x_speaker


def encode_with_rnn(rnn, x, length):

    total_length = x.size(1)
    rnn_inputs = nn.utils.rnn.pack_padded_sequence(x, length.cpu(), batch_first=True, enforce_sorted=False)
    rnn_outputs, _ = rnn(rnn_inputs)
    rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True, total_length=total_length)
    return rnn_outputs


def pool_and_pad(x, attention_mask, context_mask, seqlens, use_bert=False):

    batch_size, max_seq_len_ex, max_text_seq_len = attention_mask.shape
    dim = x.shape[-1]
    # mean pooling
    if use_bert:
        x = x[:, 0]
    else:

        mask_for_fill = attention_mask[context_mask, :].unsqueeze(-1).expand(-1, -1, dim).bool()
        x = x.masked_fill(~mask_for_fill, 0)
        x = torch.sum(x, dim=1) / torch.sum(mask_for_fill, dim=1)

    for ibatch in range(batch_size):
        fullzeropad4insert = torch.zeros([max_seq_len_ex - seqlens[ibatch], dim], device=x.device)
        index4insert = ibatch * max_seq_len_ex + seqlens[ibatch]
        x = torch.cat([x[:index4insert], fullzeropad4insert, x[index4insert:]], dim=0)
    x = x.view(batch_size, max_seq_len_ex, dim)
    return x
