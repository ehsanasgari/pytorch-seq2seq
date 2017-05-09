from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time
from dataloaders.tgif import loader
import torch

from torch import optim
from torch.nn import CrossEntropyLoss
from lstm_attention import EncoderRNN, AttnDecoderRNN, deploy, train_batch
from torch.nn.utils.rnn import pad_packed_sequence

train_loader, val_loader, vocab = loader(batch_size=8)

def sampler(x, pad_idx=1):
    def _skip_eos(row):
        s = []
        for i in row:
            if i == pad_idx:
                break
            s.append(vocab.vocab.itos[i])
        return s
    x_np = x.data.cpu().numpy().T
    return [' '.join(_skip_eos(row)) for row in x_np]

d_enc_input = 300
d_enc = 256
d_dec_input = 300
d_dec = 128

encoder = EncoderRNN(d_enc_input, d_enc, use_cnn=True)
decoder = AttnDecoderRNN(d_dec_input, d_enc*2, d_dec, vocab_size=len(vocab.vocab))
criterion = CrossEntropyLoss()

if torch.cuda.is_available():
    print("Using cuda")
    encoder.cuda()
    decoder.cuda()
    criterion.cuda()

learning_rate = 0.0001
encoder_optimizer = optim.RMSprop(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.RMSprop(decoder.parameters(), lr=learning_rate)

for epoch in range(1, 50):
    for b, (train_x, train_y, train_y_lens) in enumerate(train_loader):
        if b % 100 == 1:
            for val_b, (val_x, val_y, val_y_lens) in enumerate(val_loader):
                sampled_outs = sampler(deploy(encoder, decoder, val_x))
                targets = sampler(val_y)
                for i in range(10):
                    print("----")
                    print("Pred: {}".format(sampled_outs[i]))
                    print("Target: {}".format(targets[i]))
                    print("----", flush=True)
                break

        start = time.time()
        loss = train_batch(encoder, decoder, [encoder_optimizer, decoder_optimizer], criterion,
                           train_x, train_y, train_y_lens)
        dur = time.time() - start
        if b % 50 == 0:
            print("e{:2d}b{:3d} Loss is {}, ({:.3f} sec/batch)".format(epoch, b, loss.data[0], dur))