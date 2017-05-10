from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence
from pytorch_misc import rnn_mask, packed_seq_iter, seq_lengths_from_pad, \
    const_row
from torchvision import models

MAX_CNN_SIZE = 32


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, use_embedding=False, use_cnn=False, vocab_size=None,
                 pad_idx=None):
        """
        Bidirectional GRU for encoding sequences
        :param input_size: Size of the feature dimension (or, if use_embedding=True, the embed dim)
        :param hidden_size: Size of the GRU hidden layer. Outputs will be hidden_size*2
        :param use_embedding: True if we need to embed the sequences
        :param vocab_size: Size of vocab (only used if use_embedding=True)
        """

        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, bidirectional=True)

        self.use_embedding = use_embedding
        self.use_cnn = use_cnn
        self.vocab_size = vocab_size
        self.embed = None
        if self.use_embedding:
            assert self.vocab_size is not None
            self.pad = pad_idx
            self.embed = nn.Embedding(self.vocab_size, self.input_size, padding_idx=pad_idx)
        elif self.use_cnn:
            self.embed = models.resnet101(pretrained=True)

            # TODO
            for param in self.embed.parameters():
                param.requires_grad = False
            self.embed.fc = nn.Linear(self.embed.fc.in_features, self.input_size)

            # Init weights (should be moved.)
            self.embed.fc.weight.data.normal_(0.0, 0.02)
            self.embed.fc.bias.data.fill_(0)

    def forward(self, x):
        """
        Forward pass
        :param x: Can be a time-first PackedSequence (seq. where lengths are in descending order),
                  a T x batch_size matrix, where entries that == pad_idx are not used.

        :return: output: [batch_size, max_T, 2*hidden_size] matrix
                 lengths: [batch_size] list of the lengths
                 h_n: [batch_size, 2*hidden_size] vector of the hidden state
        """
        if isinstance(x, PackedSequence):
            x_embed = x if self.embed is None else PackedSequence(self.embed(x.data), x.batch_sizes)
        else:
            x_embed = x if self.embed is None else self.embed(x)

        output, h_n = self.gru(x_embed)
        h_n_fixed = h_n.transpose(0,1).contiguous().view(-1, self.hidden_size * 2)

        if isinstance(output, PackedSequence):
            output, lengths = pad_packed_sequence(output)
        else:
            lengths = [output.size(1)]*output.size(0)
        output_t = output.transpose(0, 1).contiguous()
        return output_t, lengths, h_n_fixed


class Attention(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_dim=None):
        """
        Attention mechanism
        :param enc_dim: Dimension of hidden states of the encoder h_j
        :param dec_dim: Dimension of the hidden states of the decoder s_{i-1}
        :param dec_dim: Dimension of the internal dimension (default: same as decoder).
        """
        super(Attention, self).__init__()

        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.attn_dim = self.dec_dim if attn_dim is None else attn_dim

        # W_h h_j
        self.encoder_in = nn.Linear(self.enc_dim, self.attn_dim, bias=False)
        self.decoder_in = nn.Linear(self.dec_dim, self.attn_dim, bias=False)
        self.att_linear = nn.Linear(self.attn_dim, 1, bias=False)

    def forward(self, dec_state, context, mask=None):
        """
        :param dec_state:  batch x dec_dim
        :param context: batch x T x enc_dim
        :return: Weighted context, batch x enc_dim
                 Alpha weights (viz), batch x T
        """
        batch, source_l, enc_dim = context.size()
        assert enc_dim == self.enc_dim

        # W*s over the entire batch (batch, attn_dim)
        dec_contrib = self.decoder_in(dec_state)

        # W*h over the entire length & batch (batch, source_l, attn_dim)
        enc_contribs = self.encoder_in(
            context.view(-1, self.enc_dim)).view(batch, source_l, self.attn_dim)

        # tanh( Wh*hj + Ws s_{i-1} )     (batch, source_l, dim)
        pre_attn = F.tanh(enc_contribs + dec_contrib.unsqueeze(1).expand_as(enc_contribs))

        # v^T*pre_attn for all batches/lengths (batch, source_l)
        energy = self.att_linear(pre_attn.view(-1, self.attn_dim)).view(batch, source_l)

        # Apply the mask. (Might be a better way to do this)
        if mask is not None:
            shift = energy.max(1)[0]
            energy_exp = (energy - shift.expand_as(energy)).exp() * mask
            alpha = torch.div(energy_exp, energy_exp.sum(1).expand_as(energy_exp))
        else:
            alpha = F.softmax(energy)

        weighted_context = torch.bmm(alpha.unsqueeze(1), context).squeeze(1)  # (batch, dim)
        return weighted_context, alpha


class AttnDecoderRNN(nn.Module):
    def __init__(self, embed_dim, encoder_hidden_dim, hidden_dim,
                 vocab_size, bos_token=0, pad_idx=1, eos_token=2):
        """
        Initializes the RNN
        :param embed_dim: Dimension of the embeddings
        :param encoder_hidden_dim: Hidden dim of the encoder, for attention purposes
        :param hidden_dim: Hidden dim of the decoder
        :param vocab_size: Number of words in the vocab
        :param bos_token: To use during decoding (non teacher forcing mode))
        :param bos: beginning of sentence token
        :param unk: unknown token (not used)
        """
        self.bos_token = bos_token
        self.embed_dim = embed_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.eos_token = eos_token

        super(AttnDecoderRNN, self).__init__()

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=self.pad_idx)
        self.gru = nn.GRU(self.embed_dim + self.encoder_hidden_dim, self.hidden_dim)
        self.attn = Attention(self.encoder_hidden_dim, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.vocab_size)

        # Differs from the paper because I'm using the final forward and backward LSTM states
        self.init_hidden = nn.Linear(self.encoder_hidden_dim, self.hidden_dim)

    def _lstm_loop(self, state, embed, context, mask=None):
        """
        :param state: Current decoder state (batch_size, dec_dim)
        :param embed: Embedding size (batch_size, embed_dim)
        :param context: All the context from encoder (batch_size, source_l, enc_dim)
        :param mask: Mask of size (batch_size, source_l) with 1 if that token is valid in encoder,
                     0 otherwise.
        :return: out: (batch_size, vocab_size) distribution over labels
                 state: (batch_size, dec_dim) next state
                alpha: (batch_size, source_l) distribution over the encoded hidden states,
                       useful for debugging maybe
        """
        c_t, alpha = self.attn(state, context, mask)
        gru_inp = torch.cat((embed, c_t), 1).unsqueeze(0)

        state = self.gru(gru_inp, state.unsqueeze(0))[0].squeeze(0)
        out = self.out(state)

        return out, state, alpha

    def _teacher_force(self, state, input_data, input_batches, context, mask):
        """
        Does teacher forcing for training

        :param state: (batch_size, dim) state size
        :param input_data: (t*batch_size) flattened array
        :param input_batches: Batch sizes for each timestep in input_data
        :param context: (T, batch_size, dim) of context
        :param mask: (T, batch_size) mask for context
        :return: Predictions (t*batch_size), exactly the same length as input_data
        """
        embeds = self.embedding(input_data)
        outputs = []
        for emb, batch_size in zip(packed_seq_iter((embeds, input_batches)),
                                   input_batches):

            out, state, alpha = self._lstm_loop(
                state[:batch_size],
                emb[:batch_size],
                context[:batch_size],
                mask[:batch_size],
            )
            outputs.append(out)
        return torch.cat(outputs)

    def _sample(self, state, context, mask, max_len=20):
        """
        Performs sampling
        """
        batch_size = state.size(0)

        toks = [const_row(self.bos_token, batch_size, volatile=True)]

        lens = torch.IntTensor(batch_size)
        if torch.cuda.is_available():
            lens = lens.cuda()

        for l in range(max_len + 1):  # +1 because of EOS
            out, state, alpha = self._lstm_loop(state, self.embedding(toks[-1]), context, mask)

            # Do argmax (since we're doing greedy decoding)
            toks.append(out.max(1)[1].squeeze(1))

            lens[(toks[-1].data == self.eos_token) & (lens == 0)] = l+1
            if all(lens):
                break
        lens[lens == 0] = max_len+1
        return torch.stack(toks, 0), lens

    def forward(self, h_cat, context, context_lens, input_data=None, max_len=20):
        """
        Does teacher forcing for training

        :param h_cat: (batch_size, d_enc*2) final state size
        :param inputs: PackedSequence (T*batch_size) of inputs
        :param context: (T, batch_size, dim) of context
        :param context_lens: (batch_size) Length of each batch
        :return:
        """
        state = self._init_hidden(h_cat)
        mask = rnn_mask(context_lens)

        if input_data is None:
            return self._sample(state, context, mask, max_len)

        if isinstance(input_data, PackedSequence):
            tf_out = self._teacher_force(state, input_data.data, input_data.batch_sizes, context, mask)
            return PackedSequence(tf_out, input_data.batch_sizes)

        # Otherwise, it's a normal torch tensor
        batch_size = input_data.size(1)
        T = input_data.size(0) - 1 # Last token is EOS

        tf_out = self._teacher_force(state, input_data[:T].view(-1), [batch_size] * T, context, mask)
        tf_out = tf_out.view(T, batch_size, -1)
        return tf_out

    def _init_hidden(self, h_dec):
        return F.tanh(self.init_hidden(h_dec))


def deploy(encoder, decoder, input_variable, max_len=20):
    """
    calls the enc/dec model
    :param input_variable: Inputs to encode
    :param encoder: EncoderRNN
    :param decoder: AttnDecoderRNN
    :param max_len: Maximum length of generated captions
    :return:
    """
    context, context_lens, final_h = encoder(input_variable)
    return decoder(final_h, context, context_lens, max_len=max_len)[0]


def train_batch(encoder, decoder, optimizers, criterion, input_variable, target_variable, lengths=None):
    """
    calls for training
    :param input_variable: Inputs to encode
    :param targets_in: <bos> padded PackedSequence of targets
    :param targets_out: <eos> ending PackedSequence of targets
    :param encoder: EncoderRNN
    :param decoder: AttnDecoderRNN
    :param encoder_optimizer:
    :param decoder_optimizer:
    :param criterion: Callable loss function for two sequences
    :return: loss (also does an update on the parameters)
    """
    for opt in optimizers:
        opt.zero_grad()

    context, context_lens, final_h = encoder(input_variable)
    outputs = decoder(final_h, context, context_lens, input_data=target_variable)

    if isinstance(outputs, PackedSequence):
        outputs, lengths = outputs

    loss = 0
    for o, l, t in zip(outputs.transpose(0,1), lengths, target_variable.t()):
        loss += criterion(o[:l], t[1:(l+1)])/(l*len(lengths))
    loss.backward()
    for opt in optimizers:
        opt.step()
    return loss
