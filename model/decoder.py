import torch
import torch.nn as nn
import numpy as np

# Hyperparametrs
embed_dim,num_heads=64,8
mlp_size=128

# Input requirements for decoder
# positional_encoding = torch.randn(32,16,64)
# encoder_output = torch.randn(32,16,64)

# LABELS
LABELS = {'\n':0,'A':1,'B':2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17, 'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': 22, 'w': 23, 'x': 24, 'y': 25,'Z':26}

# A single decoder layer
class DecoderLayer(nn.Module):
  def __init__(self,embed_dim:int=64,num_heads:int=16,batch_first=True,dropout=0.1,mlp_size:int=128):
    super().__init__()
    self.object_queries = torch.zeros(32,16,64)
    self.tgt_mask=None

    self.multiHeadAttention1 = nn.MultiheadAttention(embed_dim=embed_dim,
                                                    num_heads = num_heads,
                                                    batch_first=batch_first,
                                                    dropout = dropout)

    self.layerNorm = nn.LayerNorm(normalized_shape=embed_dim)

    self.multiHeadAttention2 = nn.MultiheadAttention(embed_dim=embed_dim,
                                                    num_heads = num_heads,
                                                    batch_first=batch_first,
                                                    dropout = dropout)

    self.mlp = nn.Sequential(
        nn.Linear(in_features=embed_dim,
                  out_features=mlp_size),
        nn.GELU(),
        nn.Dropout(p=dropout),
        nn.Linear(in_features=mlp_size,
                  out_features=embed_dim),
        nn.Dropout(p=dropout)
    )

  def forward(self,encoder_output,positional_encoding):
    mHA1 = self.multiHeadAttention1(query=self.object_queries+self.object_queries,
                              key=self.object_queries+self.object_queries,
                              value=self.object_queries,
                              need_weights=False)[0]
    ln1 = self.layerNorm(mHA1) + self.object_queries

    mHA2, _ = self.multiHeadAttention2(query=ln1+self.object_queries,
                                    key=positional_encoding+encoder_output,
                                    value=encoder_output,
                                    need_weights=False)
    ln2 = self.layerNorm(mHA2)+ln1

    mlp1 = self.mlp(ln2)
    output = self.layerNorm(mlp1)+ln2
    return output

class Decoder(nn.Module):
  def __init__(self,num_layers=4):
    super().__init__()

    self.layers = nn.ModuleList(
      [
          DecoderLayer(embed_dim=embed_dim,num_heads=num_heads,batch_first=True,dropout=0.1,mlp_size=128)
          for _ in range(num_layers)
      ]
    )

    self.linearLayer=nn.Linear(in_features=embed_dim,
                      out_features=27)

  def forward(self,encoder_output,positional_encoding):
    for layer in self.layers:
      x = layer(encoder_output,positional_encoding)

    word=[[]]
    for i in range(len(x)):
      if(i != len(x)-1):
        word.append([])
      for j in range(num_heads):
        y=self.linearLayer(x[i,j,:])
        word[i].append(y.tolist())

    return word
if __name__ =="__main__":
    decoder =Decoder()
    output=decoder(encoder_output,positional_encoding)
    output=torch.tensor(output)
    print(output.shape)
