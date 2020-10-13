# coding: utf-8

import torch
import torch.nn as nn
from torch import Tensor

from torch_geometric.nn import GlobalAttention,GatedGraphConv,TopKPooling
from torch_geometric.data import Data,Batch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from joeynmt.embeddings import Embeddings
from joeynmt.constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN

from joeynmt.helpers import freeze_params
from joeynmt.transformer_layers import TransformerEncoderLayer,PositionalEncoding
from joeynmt.encoders import Encoder
from joeynmt.vocabulary import Vocabulary
import torch.nn.functional as F
import pdb


class GraphEncoder(Encoder):
    """
    Dummy encoder based on Transformer Encoder
    """

    """Encodes a sequence of word embeddings"""
    #pylint: disable=unused-argument
    def __init__(self,
                 hidden_size: int = 1,
                 emb_size: int = 1,
                 num_layers: int = 1,
                 dropout: float = 0.,
                 emb_dropout: float = 0.,
                 freeze: bool = False,
                 edge_vocab: Vocabulary = None,
                 cfg: dict = None,
                 **kwargs) -> None:
        """
        Create a new recurrent encoder.
        :param hidden_size: Size of each RNN.
        :param emb_size: Size of the word embeddings.
        :param num_layers: Number of encoder RNN layers.
        :param dropout:  Is applied between RNN layers.
        :param emb_dropout: Is applied to the RNN input (word embeddings).
        :param bidirectional: Use a bi-directional RNN.
        :param freeze: freeze the parameters of the encoder during training
        :param kwargs:
        """

        super(GraphEncoder, self).__init__()

        self.emb_dropout = torch.nn.Dropout(p=emb_dropout, inplace=False)
        self.emb_size = emb_size
        self.num_layers=num_layers
        self.hidden_size=hidden_size
        self.edge_vocab=edge_vocab


        ###
        #New embeddings for the edges
        ###
        self.edge_embeddings= Embeddings(
            **cfg["embeddings"], vocab_size=len(edge_vocab),
            padding_idx=edge_vocab.stoi[PAD_TOKEN])


        self.gate_nn = Seq(Lin(hidden_size, hidden_size), ReLU(), Lin(hidden_size, 1))
        self.gAtt = GlobalAttention(self.gate_nn)


        self.ggnn = GatedGraphConv(
            hidden_size, num_layers)

        self._output_size = hidden_size

        if freeze:
            freeze_params(self)

    #pylint: disable=invalid-name, unused-argument
    def _check_shapes_input_forward(self, embed_src: Tensor, src_length: Tensor,
                                    mask: Tensor) -> None:
        """
        Make sure the shape of the inputs to `self.forward` are correct.
        Same input semantics as `self.forward`.

        :param embed_src: embedded source tokens
        :param src_length: source length
        :param mask: source mask
        """
        assert embed_src.shape[0] == src_length.shape[0]
        assert embed_src.shape[2] == self.emb_size
       # assert mask.shape == embed_src.shape
        assert len(src_length.shape) == 1

    #pylint: disable=arguments-differ
    def forward(self, embed_src: Tensor, src_length: Tensor, mask: Tensor,
                batch: Batch) \
            -> (Tensor, Tensor):
        """
        Applies a bidirectional RNN to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].

        :param embed_src: embedded src inputs,
            shape (batch_size, src_len, embed_size)
        :param src_length: length of src inputs
            (counting tokens before padding), shape (batch_size)
        :param mask: indicates padding areas (zeros where padding), shape
            (batch_size, src_len, embed_size)
        :return:
            - output: hidden states with
                shape (batch_size, max_length, directions*hidden),
            - hidden_concat: last hidden state with
                shape (batch_size, directions*hidden)
        """
        self._check_shapes_input_forward(embed_src=embed_src,
                                         src_length=src_length,
                                         mask=mask)

        # apply dropout to the emmbeding input
        embed_src = self.emb_dropout(embed_src)
        embed_edges = self.edge_embeddings(batch.edge)
        embed_edges = self.emb_dropout(embed_edges)
        embeddings= torch.cat((embed_src,embed_edges),dim=1)
        pdb.set_trace()
        data=self.reorder_edges_words(embed_src,batch)
        pdb.set_trace()
        x, edge_index, batch = data.x.cuda(), data.edge_index.cuda(), data.batch.cuda()
        #pdb.set_trace()
        x = F.relu(self.ggnn(x, edge_index))
        #x= self.pool1(x, x)
        #pdb.set_trace()
        hidden_concat=self.gAtt(x,data.batch.cuda())
        output=x.view((embed_src.shape[0],embed_src.shape[1],-1))
        #hidden_concat=torch.zeros((batch_size,self.hidden_size))
        #hidden_concat=self.gAtt(x,data.batch)
        return output, hidden_concat
    
    def reorder_edges_words(self,embed,batch):
        """
        Input: batch with edge_org,edge_trg, and x
        return Batch of pytorch geometric
        """
        data_list=[]
        orgs=[]
        trgs=[]
        max_lenght=int(batch.src_lengths[0])
        for i,edge_orgs in enumerate(batch.edge_org):
            curr_lenght=int(batch.src_lengths[0])
            offset=max_lenght-curr_lenght
            for j,edge_org in enumerate(edge_orgs): 
                org=edge_org
                trg=batch.edge_trg[i][j]
                if int(org)!=0 and int(trg)!=0:
                    if int(org)>curr_lenght:
                        orgs.append(int(org)+offset)
                    else:
                        orgs.append(int(org))
                    if int(trg)>curr_lenght:
                        trgs.append(int(trg)+offset)
                    else:
                        trgs.append(int(trg))

            data_list.append(Data(embed[i],torch.tensor([orgs,trgs],dtype=torch.long)))

        return Batch.from_data_list(data_list)
    def reorder_pes(self,x):
        pass
    def create_simple_edges(self,num_sentences,len_sentences):
      final_edges=[]
      for i in range(num_sentences):
        if i==0:
          sentence_edges=list(range(len_sentences))
        else:
          sentence_edges=list(range((len_sentences*i),len_sentences*(i+1)))
    
        final_edges=final_edges+sentence_edges[1:]+[sentence_edges[0]]
      return range(num_sentences*len_sentences),final_edges

