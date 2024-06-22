import copy
import torch

from torch import nn

from models.embedding.token_embedding import TokenEmbedding
from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.transformer_embedding import TransformerEmbedding

from models.layer.multi_head_attention import MultiHeadAttention
from models.layer.position_wise_feed_forward import PositionWiseFeedForward
from models.layer.encoder_layer import EncoderLayer
from models.layer.decoder_layer import DecoderLayer

from models.model.encoder import Encoder
from models.model.decoder import Decoder
from models.model.transformer import Transformer

def build_model(src_vocab_size,
                trg_vocab_size,
                max_seq_len = 256,
                d_embed = 512,
                num_layer = 6,
                d_model = 512,
                num_heads = 8,
                d_ff = 2048,
                dr_rate = 0.1,
                norm_eps = 1e-5,
                device=torch.device("cpu")):
    
    ## Embedding
    src_token_embedder = TokenEmbedding(d_embed=d_embed, vocab_size=src_vocab_size)
    trg_token_embedder = TokenEmbedding(d_embed=d_embed, vocab_size=trg_vocab_size)
    pos_encoder = PositionalEncoding(d_embed=d_embed, max_seq_len=max_seq_len, device=device)

    src_embedder = TransformerEmbedding(token_embedder=src_token_embedder, position_encoder=pos_encoder, dr_rate=dr_rate)
    trg_embedder = TransformerEmbedding(token_embedder=trg_token_embedder, position_encoder=pos_encoder, dr_rate=dr_rate)

    ## Multi-Head Attention : qkv_fc, out_fc는 nn.Linear를 호출시 전달함으로써 encoder_layers와 decoder_layers가 해당 층들을 재사용하는 것과 같음.
    attention_layer = MultiHeadAttention(d_model=d_model, 
                                         num_heads=num_heads, 
                                         qkv_fc=nn.Linear(d_embed, d_model), 
                                         out_fc=nn.Linear(d_model, d_embed), 
                                         dr_rate=dr_rate)

    ## Position-wise FeedForward
    position_ff_layer = PositionWiseFeedForward(fc1=nn.Linear(d_embed, d_ff), fc2=nn.Linear(d_ff, d_embed), dr_rate=dr_rate)

    ## LayerNorm
    norm_layer = nn.LayerNorm(d_embed, eps=norm_eps)

    ## Encoder
    encoder_layer = EncoderLayer(self_attention=copy.deepcopy(attention_layer), 
                                 position_ff=copy.deepcopy(position_ff_layer),
                                 norm_layer=copy.deepcopy(norm_layer),
                                 dr_rate=dr_rate)
    
    encoder = Encoder(encoder_layer=encoder_layer, num_layer=num_layer, norm_layer=copy.deepcopy(norm_layer))
    
    ## Decoder
    decoder_layer = DecoderLayer(self_attention=copy.deepcopy(attention_layer), 
                                 cross_attention=copy.deepcopy(attention_layer),
                                 position_ff=copy.deepcopy(position_ff_layer),
                                 norm_layer=copy.deepcopy(norm_layer),
                                 dr_rate=dr_rate)
    
    decoder = Decoder(decoder_layer=decoder_layer, num_layer=num_layer, norm_layer=copy.deepcopy(norm_layer))

    ## Transformer Outpyt Layer
    output_layer = nn.Linear(d_model, trg_vocab_size)

    ## Transformer Model
    model = Transformer(src_embed=src_embedder, trg_embed=trg_embedder, encoder=encoder, decoder=decoder, generator=output_layer).to(device)
    model.device = device

    return model