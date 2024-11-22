from turtle import forward
import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self,dim,num_classes=4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim,num_classes)
    
    def forward(self,x,context_mask):
        x = self.head(self.norm(x))
        return x