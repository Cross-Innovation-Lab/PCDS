from turtle import forward
import os
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Config, HubertModel, HubertConfig, WavLMModel, WavLMConfig, AutoModel
from .utils import ScalarMix


class Speech_Pretrain_Model(nn.Module):
    def __init__(self, dim, pretrain='hubert', finetune=False) -> None:
        super().__init__()
        self.finetune = finetune
        configmap = {
            'mask_time_prob': 0.08,
            'mask_time_length': 15,
            'mask_feature_prob': 0.05,
            'mask_feature_length': 64
        }
        assert pretrain in ['hubert', 'wav2vec2', 'wavlm'], "Unkown pretrain model for finetuning"
        if finetune:
            if pretrain == 'hubert':
                bundle = torchaudio.pipelines.HUBERT_BASE
                self.pretrain = bundle.get_model()
            elif pretrain == 'wav2vec2':
                bundle = torchaudio.pipelines.WAV2VEC2_BASE
                self.pretrain = bundle.get_model()
        else:
            if pretrain == 'hubert':
                # config = HubertConfig.from_pretrained("facebook/hubert-base-ls960", output_hidden_states=~finetune)
                # config.update(configmap)
                # self.pretrain = HubertModel.from_pretrained("facebook/hubert-base-ls960", config=config)
                config = HubertConfig.from_pretrained("/public1/home/stu52205901024/hugging_hub/hubert/config.json",
                                                        output_hidden_states=~finetune)
                config.update(configmap)
                self.pretrain = HubertModel.from_pretrained("/public1/home/stu52205901024/hugging_hub/hubert",
                                                            local_files_only=True,
                                                            config=config)

            elif pretrain == 'wav2vec2':
                # config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base", output_hidden_states=~finetune)
                # config.update(configmap)
                # self.pretrain = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", config=config)
                config = Wav2Vec2Config.from_pretrained("/public1/home/stu52205901024/hugging_hub/wav2vec2/config.json",
                                                        output_hidden_states=~finetune)
                config.update(configmap)
                self.pretrain = Wav2Vec2Model.from_pretrained("/public1/home/stu52205901024/hugging_hub/wav2vec2",
                                                            local_files_only=True,
                                                            config=config)
            elif pretrain == 'wavlm':
                # config = WavLMConfig.from_pretrained("microsoft/wavlm-base", output_hidden_states=~finetune)
                # config.update(configmap)
                # self.pretrain = WavLMModel.from_pretrained("microsoft/wavlm-base", config=config)
                config = WavLMConfig.from_pretrained("/public1/home/stu52205901024/hugging_hub/wavlm/config.json",
                                                        output_hidden_states=~finetune)
                config.update(configmap)
                self.pretrain = WavLMModel.from_pretrained("/public1/home/stu52205901024/hugging_hub/wavlm",
                                                            local_files_only=True,
                                                            config=config)
            self.weight_audio = ScalarMix(13)
            self.weight_speaker = ScalarMix(13)


    def forward(self, x):
        '''
        x : seq_num, seq_len
        '''
        if self.finetune:
            # fintune with weighted-sum
            # x = self.pretrain(x).last_hidden_state # huggingface
            with torch.no_grad():
                x, _ = self.pretrain.feature_extractor(x, None)
            x = self.pretrain.encoder(x, None)
        else:
            # feature extraction with weighted-sum
            self.pretrain.eval()
            with torch.no_grad():
                x = self.pretrain(x).hidden_states
            # x = self.weight(x)
            # print("hidden_states:",x)
            x_audio = self.weight_audio(x)
            # print("x_audio:", x_audio)
            x_speaker = self.weight_speaker(x)
        return x_audio, x_speaker
    
class Text_Pretrain_Model(nn.Module):
    def __init__(self, finetune=False) -> None:
        super().__init__()
        self.finetune = finetune
        self.pretrain = AutoModel.from_pretrained("/public1/home/stu52205901024/hugging_hub/simcse",
                                                    local_files_only=True)

    def forward(self, x, mask=None):
        '''
        x : seq_num, seq_len
        mask : seq_num, seq_len
        '''
        if self.finetune:
            x = self.pretrain(x, mask).last_hidden_state
        else:
            self.pretrain.eval()
            with torch.no_grad():
                x = self.pretrain(x, mask).last_hidden_state
        
        return x