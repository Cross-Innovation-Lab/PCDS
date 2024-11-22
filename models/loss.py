import torch
import torch.nn as nn


# class SupConLoss(nn.Module):

#     def __init__(self, temperature=0.07, contrast_mode='base', K=256) -> None:
#         '''
#         contrast_mode: Type of contrast modes in "base", "queue".
#         K: size of queue, note the shape of queue is [K, dim]
#         '''
#         super().__init__()
#         self.temperature = temperature
#         self.contrast_mode = contrast_mode
#         self.K = K
#         if self.contrast_mode == 'queue':
#             self.text_queue = None
#             self.audio_queue = None
#             self.label_queue = None
#             self.queue_empty = True

#     def dequeue_and_enqueue(self, text_feature, audio_feature, labels):
#         if self.queue_empty:
#             self.text_queue = text_feature.clone().detach()
#             self.audio_queue = audio_feature.clone().detach()
#             self.label_queue = labels.clone().detach()
#             self.queue_empty = False
#         else:
#             self.text_queue = torch.cat((self.text_queue, text_feature.clone()), dim=0)
#             self.audio_queue = torch.cat((self.audio_queue, audio_feature.clone()), dim=0)
#             self.label_queue = torch.cat((self.label_queue, labels.clone()), dim=0)
#         if self.label_queue.shape[0] > self.K:
#             self.text_queue = self.text_queue[-self.K:, :]
#             self.audio_queue = self.audio_queue[-self.K:, :]
#             self.label_queue = self.label_queue[-self.K:]

#     def compute_loss(self, anchor_feature, contrast_feature, anchor_labels, contrast_labels):
#         '''
#         anchor_feature: seq_num, dim
#         contrast_feature: seq_num, dim
#         labels: seq_num
#         '''
#         device = (torch.device('cuda') if anchor_feature.is_cuda else torch.device('cpu'))
        
#         seq_num = anchor_labels.shape[0]
#         mask = torch.eq(anchor_labels.unsqueeze(-1),
#                         contrast_labels.unsqueeze(0)).float().to(device)  # [seq_num, seq_num]

#         anchor_cnt = contrast_cnt = 1

#         # if self.contrast_mode == 'one':
#         #     anchor_feature = features[:, 0]
#         #     anchor_cnt = 1
#         # elif self.contrast_mode == 'all':
#         #     anchor_feature = contrast_feature
#         #     anchor_cnt = contrast_cnt

#         # compute logits
#         anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.mT),
#                                         self.temperature)  # [seq_num, seq_num]
#         # for numerical stability
#         logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
#         logits = anchor_dot_contrast - logits_max.detach()  # [seq_num, seq_num]

#         # tile mask
#         mask = mask.repeat(anchor_cnt, contrast_cnt)
#         # mask-out self-contrast cases
#         # logits_mask = torch.scatter(torch.ones_like(mask),1,torch.arange(contrast_cnt*seq_num).view(-1, 1).to(device),0)
#         # mask = mask * logits_mask

#         # compute log_prob
#         exp_logits = torch.exp(logits) * mask  # [seq_num, seq_num]
#         log_prob = logits - torch.log(exp_logits.sum(1, True))

#         # compute mean of log-likelihood over positive
#         mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
#         # compute loss
#         # loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
#         loss = -mean_log_prob_pos
#         loss = loss.view(anchor_cnt, seq_num).mean()

#         return loss

#     def forward(self, text_feature, audio_feature, labels):
#         '''
#         text_feature: seq_num, dim
#         audio_feature: seq_num, dim
#         labels: seq_num
#         '''
#         if self.contrast_mode == 'queue':
#             self.dequeue_and_enqueue(text_feature, audio_feature, labels)
#             device = (torch.device('cuda') if text_feature.is_cuda else torch.device('cpu'))
#             text_loss = self.compute_loss(text_feature,
#                                           self.audio_queue.clone().detach().to(device), labels,
#                                           self.label_queue.clone().detach().to(device))
#             audio_loss = self.compute_loss(audio_feature,
#                                            self.text_queue.clone().detach().to(device), labels,
#                                            self.label_queue.clone().detach().to(device))
#             # text_loss = self.compute_loss(text_feature,
#             #                               self.audio_queue.clone().to(device), labels,
#             #                               self.label_queue.clone().to(device))
#             # audio_loss = self.compute_loss(audio_feature,
#             #                                self.text_queue.clone().to(device), labels,
#             #                                self.label_queue.clone().to(device))
#             loss = 0.5 * text_loss + 0.5 * audio_loss
#         elif self.contrast_mode == 'base':
#             text_loss = self.compute_loss(text_feature, audio_feature, labels, labels)
#             audio_loss = self.compute_loss(audio_feature, text_feature, labels, labels)
#             loss = 0.5 * text_loss + 0.5 * audio_loss
#         return loss


    

class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07, contrast_mode='one', base_temperature=0.07) -> None:
        """
        contast_mode: 'all' for contrast all the features, 'one' for contrast bwtween views
        
        """
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        # self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        '''
        features: seq_num, n_view, dim
        labels: seq_num
        '''
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        seq_num = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(seq_num,dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device) # [seq_num, seq_num]
        elif mask is not None:
            mask =mask.float().to(device)

        
        contrast_feature = torch.cat(torch.unbind(features, 1), 0)  

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0] # [seq_num, dim]
            anchor_cnt = 1
            contrast_feature = features[:, 1] # [seq_num, dim]
            contrast_cnt = 1
        elif self.contrast_mode == 'all':
            anchor_feature = torch.cat(torch.unbind(features, 1), 0) # [seq_num, dim]
            anchor_cnt = features.shape[1]
            contrast_feature = torch.cat(torch.unbind(features, 1), 0) # [seq_num, dim]
            contrast_cnt = features.shape[1]

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),
                                        self.temperature)  # [seq_num, seq_num]
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # [seq_num, seq_num]

        # tile mask
        mask = mask.repeat(anchor_cnt, contrast_cnt)
        # whethter to mask-out self-contrast cases
        if self.contrast_mode == 'one':
            logits_mask = torch.ones_like(mask).float().to(device)
        elif self.contrast_mode == 'all':
            logits_mask = torch.scatter(torch.ones_like(mask),1,torch.arange(anchor_cnt*seq_num).view(-1, 1).to(device),0)
            mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask # [seq_num, seq_num]
        log_prob = logits - torch.log(exp_logits.sum(1, True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # compute loss
        loss = - mean_log_prob_pos
        loss = loss.view(anchor_cnt,seq_num).mean()

        return loss