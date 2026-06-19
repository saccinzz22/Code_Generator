import torch # type: ignore
import torch.nn as nn # type: ignore

class SelfAttention(nn.Module):
    def __init__(self, embed_size, head):
        super().__init__()
        self.head=head
        self.embed_size=embed_size
        self.head_dim=embed_size//head
        assert(self.head_dim*head==self.embed_size), "Embed size is not divisible by no of heads"
        
        self.values=nn.Linear(self.head_dim,self.head_dim,bias=False)
        self.keys=nn.Linear(self.head_dim,self.head_dim,bias=False)
        self.queries=nn.Linear(self.head_dim,self.head_dim,bias=False)
        self.fc_out=nn.Linear(embed_size,embed_size)
        
    def forward(self, queries, keys, values, mask):
        #queries shape: [batch size, sequence length, embedding size]
        #key shape: [batch size, sequence length, embedding size]
        N=queries.shape[0] #batch size
        value_len,key_len,query_len=values.shape[1],keys.shape[1],queries.shape[1] #no of tokens
        
        #Split embedding into self.head pieces
        values=values.reshape(N,value_len,self.head,self.head_dim)
        queries=queries.reshape(N,query_len,self.head,self.head_dim)
        keys=keys.reshape(N,key_len,self.head,self.head_dim)
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        energy=torch.einsum("nqhd,nkhd->nhqk",[queries,keys]) #Attention scores(product of queries and keys summed over dimension)
        #queries shape: [N, query_len, head, head_dim]
        #keys shape: [N, key_len, head, head_dim]
        #energy shape: [N,head,query_len,key_len]
        
        if mask is not None:
            energy=energy.masked_fill(~mask,float("-1e20"))
            
        attention=torch.softmax(energy/ (self.head_dim ** 0.5),dim=3)
        #attention shape: [N, head, query_len, key_len]
        #values shape: [N,value_len, head, head_dim]
        #l=key_len=value_len
        
        out=torch.einsum("nhql,nlhd->nqhd",[attention, values])
        out=out.reshape(N, query_len, self.embed_size)
        #out shape: [N,query_len, embed_size]
        
        out=self.fc_out(out)
        return out
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, head, dropout, forward_expansion):
        #forward_expansion increases dimensionality for complex processes
        #dropout randomly sets values to zero to prevent overfitting
        super(TransformerBlock,self).__init__()
        self.attention=SelfAttention(embed_size, head)
        
        self.norm1=nn.LayerNorm(embed_size)
        self.norm2=nn.LayerNorm(embed_size)
        
        self.feed_forward=nn.Sequential(
            nn.Linear(embed_size, embed_size * forward_expansion),
            nn.GELU(),
            nn.Linear(embed_size * forward_expansion, embed_size) 
        )
        self.dropout=nn.Dropout(dropout)
    def forward(self, value, query, key, mask):
        attention=self.attention(query, key,value, mask)
        x=self.dropout(self.norm1(attention+query))
        
        forward=self.feed_forward(x)
        out=self.dropout(self.norm2(x+forward))
        return out

class Encoder(nn.Module):
    def __init__(self,
                embed_size,
                src_vocab_size,
                max_length,
                device,
                head,
                dropout,
                forward_expansion,
                num_layers
    ): 
        super(Encoder, self).__init__()
        self.embed_size=embed_size
        self.device=device
        self.word_embedding=nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding=nn.Embedding(max_length, embed_size)
        
        self.layers=nn.ModuleList(
            [
                TransformerBlock(embed_size,
                                head,
                                dropout=dropout,
                                forward_expansion=forward_expansion)
                for i in range(num_layers)
            ]
        )
        self.dropout=nn.Dropout(dropout)
    def forward(self, x, mask):
        N, seq_length=x.shape
        position = torch.arange(seq_length,device=self.device).unsqueeze(0).expand(N,seq_length)
        out= self.dropout(self.word_embedding(x) *(self.embed_size ** 0.5) + self.position_embedding(position))
        for layer in self.layers:
            out=layer(out, out, out, mask)
        return out
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, head, dropout, forward_expansion, device):
        super(DecoderBlock, self).__init__()
        self.attention=SelfAttention(embed_size, head)
        self.norm= nn.LayerNorm(embed_size)
        self.dropout=nn.Dropout(dropout)
        self.transformer_block=TransformerBlock(embed_size, head, dropout, forward_expansion)
    def forward(self, x, key, value, src_mask, trg_mask):
        attention=self.attention(x, x, x, trg_mask)
        query=self.dropout(self.norm(attention + x))
        out= self.transformer_block(value, query, key, src_mask)
        return out
class Decoder(nn.Module):
    def __init__(self,
                embed_size,
                head,
                num_layers,
                trg_vocab_size,
                forward_expansion,
                dropout,
                device,
                max_length):
        super(Decoder, self).__init__()
        self.embed_size=embed_size
        self.device=device
        self.word_embedding=nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding=nn.Embedding(max_length, embed_size)
        
        self.layers=nn.ModuleList(
            [
                DecoderBlock(embed_size, head, dropout, forward_expansion, device)
                for i in range(num_layers)
            ]
        )
        self.dropout=nn.Dropout(dropout)
        self.fc_out=nn.Linear(embed_size, trg_vocab_size)
    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length= x.shape
        position = torch.arange(seq_length,device=self.device).unsqueeze(0).expand(N,seq_length)
        x=self.dropout(self.word_embedding(x) * (self.embed_size ** 0.5) + self.position_embedding(position))
        
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        out=self.fc_out(x)
        return out

class Transformer(nn.Module):
    def __init__(self,
                src_vocab_size,
                trg_vocab_size,
                src_pad_idx,
                trg_pad_idx,
                embed_size=256,
                head=8,
                forward_expansion=4,
                dropout=0.1,
                num_layers=6,
                src_max_length=768,
                trg_max_length=512,
                device=None):
        super(Transformer, self).__init__()
        self.encoder=Encoder(embed_size,
                            src_vocab_size,
                            src_max_length,
                            device,
                            head,
                            dropout,
                            forward_expansion,
                            num_layers)
        self.decoder=Decoder(embed_size,
                            head, 
                            num_layers, 
                            trg_vocab_size, 
                            forward_expansion,
                            dropout, 
                            device,
                            trg_max_length)
        
        self.src_pad_idx=src_pad_idx
        self.trg_pad_idx=trg_pad_idx
        self.device=device
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx)\
                        .unsqueeze(1)\
                        .unsqueeze(2)
        return src_mask.to(self.device)
    
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        pad_mask = (
            trg != self.trg_pad_idx
        ).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(
            torch.ones(
                (trg_len, trg_len),
                dtype=torch.bool,
                device=self.device
            )
        )
        trg_mask = pad_mask & causal_mask
        return trg_mask.to(self.device)
    
    def forward(self, src, trg):
        src_mask= self.make_src_mask(src)
        trg_mask=self.make_trg_mask(trg)
        enc_src=self.encoder(src, src_mask)
        out=self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
    




        
        

    