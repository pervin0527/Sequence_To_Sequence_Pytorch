import copy
import torch

from torch import nn

from models.embedding.token_embedding import WordEmbedding
from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.transformer_embedding import TransformerEmbedding

from models.layer.multi_head_attention import MultiHeadAttentionLayer
from models.layer.position_wise_feed_forward import FeedForwardLayer
from models.layer.encoder_layer import EncoderBlock
from models.layer.decoder_layer import DecoderBlock

from models.model.encoder import Encoder
from models.model.decoder import Decoder
from models.model.transformer import Transformer

def build_model(src_vocab_size,
                trg_vocab_size, 
                max_seq_len=256,
                d_model=512, 
                num_layer=6, 
                num_heads=8, 
                d_ff=2048, 
                norm_eps=1e-5,
                drop_prob=0.1, 
                device=torch.device("cpu")):

    ## Word Embedding.
    src_token_embed = WordEmbedding(d_model=d_model, vocab_size=src_vocab_size)
    trg_token_embed = WordEmbedding(d_model=d_model, vocab_size=trg_vocab_size)

    ## Positional Encoding.
    src_pos_embedd = PositionalEncoding(d_model=d_model, max_seq_len=max_seq_len, device=device)
    trg_pos_embedd = PositionalEncoding(d_model=d_model, max_seq_len=max_seq_len, device=device)

    ## Word Embedding + Positional Encoding.
    trg_embed = TransformerEmbedding(word_embedding_layer=trg_token_embed, positional_encoding_layer=trg_pos_embedd, drop_prob=drop_prob)
    src_embed = TransformerEmbedding(word_embedding_layer=src_token_embed, positional_encoding_layer=src_pos_embedd, drop_prob=drop_prob)

    ## Multi-Head Self Attention.
    encoder_attention = MultiHeadAttentionLayer(d_model=d_model, num_heads=num_heads, drop_prob=drop_prob)
    decoder_attention = MultiHeadAttentionLayer(d_model=d_model, num_heads=num_heads, drop_prob=drop_prob)
    
    ## Position-Wise FeedForward.
    encoder_position_ff = FeedForwardLayer(d_model, d_ff, drop_prob=drop_prob)
    decoder_position_ff = FeedForwardLayer(d_model, d_ff, drop_prob=drop_prob)

    ## Add & Norm.
    encoder_norm = nn.LayerNorm(d_model, eps=norm_eps)
    decoder_norm = nn.LayerNorm(d_model, eps=norm_eps)

    ## Encoder Block.
    encoder_block = EncoderBlock(self_attention=encoder_attention,
                                 ffn=encoder_position_ff,
                                 norm=encoder_norm,
                                 drop_prob=drop_prob)
    
    ## Decoder Block
    decoder_block = DecoderBlock(self_attention=decoder_attention,
                                 cross_attention=decoder_attention,
                                 ffn=decoder_position_ff,
                                 norm=decoder_norm,
                                 drop_prob=drop_prob)

    ## Encoder(Encoder Block * Num_layers)
    encoder = Encoder(encoder_block=encoder_block, num_layer=num_layer, norm=encoder_norm)

    ## Decoder(Decoder Block * Num_layers)
    decoder = Decoder(decoder_block=decoder_block, num_layer=num_layer, norm=decoder_norm)

    ## Output Layer.
    generator = nn.Linear(d_model, trg_vocab_size)

    model = Transformer(src_embed=src_embed,
                        trg_embed=trg_embed,
                        encoder=encoder,
                        decoder=decoder,
                        generator=generator).to(device)
    
    model.device = device

    return model