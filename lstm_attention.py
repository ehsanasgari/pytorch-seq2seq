from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence
import numpy as np
from pytorch_misc import rnn_mask, packed_seq_iter, pad_unsorted_sequence
from torchvision import models

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, use_embedding=False, vocab_size=None):
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
        self.vocab_size = vocab_size
        if self.use_embedding:
            assert self.vocab_size is not None
            self.embed = nn.Embedding(self.vocab_size, self.input_size)
        elif self.use_cnn:
            self.cnn = models.resnet101(pretrained=True)

            # FOR NOW
            for param in self.cnn.parameters():
                param.requires_grad = False
            self.cnn.fc = nn.Linear(self.cnn.fc.in_features, self.input_size)

            # Init weights (should be moved.)
            self.cnn.fc.weight.data.normal_(0.0, 0.02)
            self.cnn.fc.bias.data.fill_(0)

    def forward(self, x, lengths=None):
        """
        Forward pass.
        :param x: Can either be a time-first packed variable length sequence, with no final
                  dimension (if using embed) or a final dimension of input_size

                  Or, if all lengths are the same, must be (T, batch_size), if embed, or
                  (T, batch_size, input_size) otherwise.

        :return: Full context: of shape (batch_size, T, hidden_size). Transposed for efficiency
                 lengths: of length batch_size, per timestep
                 hidden: Hidden representation at time t (fwd) and 1 (backward)
        """
        perm = None
        if isinstance(x, PackedSequence):
        # Assume x is a (T*batch_size,:) packed_sequence
            if self.use_embedding:
                x_data = self.embed(x.data)
            elif self.use_cnn:
                x_data = self.cnn(x.data)
            else:
                x_data = x.data

            x, lengths = pad_packed_sequence(PackedSequence(x_data, x.batch_sizes))
        elif torch.is_tensor(x) and lengths is not None:
            if self.use_embedding:
                x_data = self.embed(x)
            elif self.use_cnn:
                x_data = self.cnn(x)
            else:
                x_data = x
            x, lengths, perm = pad_unsorted_sequence(x_data, lengths)
        else:
            if not torch.is_tensor(x):
                raise ValueError('Input to EncoderRNN is not a tensor, list, or PackedSequence')

            assert x.ndimension() > 1, "Non PackedSequence input to EncoderRNN must have >= 2 dims"
            if self.use_embedding:
                new_size = list(x.size()) + [-1]
                x = self.embed(x.view(-1)).view(*new_size)
            elif self.use_cnn:
                raise NotImplementedError('not implemented yet.')

            lengths = [x.size(0) for x in range(x.size(1))]

        output, h_n = self.gru(x)

        output_t = output.transpose(0,1)
        h_n_fixed = h_n.transpose(0,1)
        if perm is not None:
            output_t = output[perm]
            h_n_fixed = h_n_fixed[perm]

        output_t = output_t.contiguous()
        h_n_fixed = h_n_fixed.contiguous().view(-1, self.hidden_size * 2)

        return output_t, lengths, h_n_fixed


class GlobalAttention(nn.Module):
    """
    From https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/GlobalAttention.py
    """
    def __init__(self, enc_dim, dec_dim, attn_dim=None):
        """
        Attention mechanism
        :param enc_dim: Dimension of hidden states of the encoder h_j
        :param dec_dim: Dimension of the hidden states of the decoder s_{i-1}
        :param dec_dim: Dimension of the internal dimension (default: same as decoder).
        """
        super(GlobalAttention, self).__init__()

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
                 vocab_size, eos=0, bos=1, unk=2):
        """
        Initializes the RNN
        :param embed_dim: Dimension of the embeddings
        :param encoder_hidden_dim: Hidden dim of the encoder, for attention purposes
        :param hidden_dim: Hidden dim of the decoder
        :param vocab_dim: Number of words in the vocab
        :param eos: end of sentence token
        :param bos: beginning of sentence token
        :param unk: unknown token (not used)
        """
        self.embed_dim = embed_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.eos = eos
        self.bos = bos
        self.unk = unk

        super(AttnDecoderRNN, self).__init__()

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=self.eos)
        self.gru = nn.GRU(self.embed_dim + self.encoder_hidden_dim, self.hidden_dim)
        self.attn = GlobalAttention(self.encoder_hidden_dim, self.hidden_dim)
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

    def sampler(self, init_h, context, context_lens, max_len=20):
        """
        Simple greedy decoding
        :param init_state:
        :param context:
        :param max_len:
        :return:
        """
        batch_size = init_h.size(0)
        input_tok = Variable(torch.LongTensor([self.bos] * batch_size))
        if torch.cuda.is_available():
            input_tok = input_tok.cuda()

        state = self._init_hidden(init_h)
        mask = rnn_mask(context_lens)
        outs = []

        has_seen_eos = np.zeros(batch_size, dtype=bool)
        for l in range(max_len+1): #+1 because of EOS
            out, state, alpha = self._lstm_loop(state, self.embedding(input_tok), context, mask)

            # Do argmax (since we're doing greedy decoding)
            input_tok = out.max(1)[1].squeeze(1)
            outs.append(input_tok)

            has_seen_eos |= (input_tok.cpu().data.numpy() == self.eos)
            if np.all(has_seen_eos):
                break
        return torch.stack(outs, 0)

    def forward(self, h_cat, inputs, context, context_lens):
        """
        Does teacher forcing for training

        :param h_cat: (batch_size, d_enc*2) final state size
        :param context: (T, batch_size, dim) of context
        :param context_lens: (batch_size) Length of each batch
        :param inputs: PackedSequence (T*batch_size) of inputs
        :return:
        """
        state = self._init_hidden(h_cat)

        embeds = self.embedding(inputs.data)
        mask = rnn_mask(context_lens)

        outputs = []
        for emb, batch_size in zip(packed_seq_iter((embeds, inputs.batch_sizes)),
                                   inputs.batch_sizes):

            out, state, alpha = self._lstm_loop(
                state[:batch_size],
                emb[:batch_size],
                context[:batch_size],
                mask[:batch_size],
            )
            outputs.append(out)
        outputs = PackedSequence(torch.cat(outputs), inputs.batch_sizes)
        return outputs

    def _init_hidden(self, h_dec):
        return F.tanh(self.init_hidden(h_dec))


def deploy_txt(input_variable, encoder, decoder, max_len=20):
    """
    calls the enc/dec model
    :param input_variable: Inputs to encode
    :param encoder: EncoderRNN
    :param decoder: AttnDecoderRNN
    :param max_len: Maximum length of generated captions
    :return:
    """
    context, context_lens, final_h = encoder(input_variable)
    return decoder.sampler(final_h, context, context_lens, max_len)


def train_batch_txt(input_variable, targets_in, targets_out, encoder, decoder,
                    encoder_optimizer, decoder_optimizer, criterion):
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
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    context, context_lens, final_h = encoder(input_variable)
    outputs = decoder(final_h, targets_in, context, context_lens)

    # NOTE: currently this is weighting longer sequences more than shorter ones.
    # This seems easier, anyone is welcome to change this though
    loss = criterion(outputs.data, targets_out.data) / len(targets_out.data)

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss
