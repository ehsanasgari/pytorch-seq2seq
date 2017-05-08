"""
German to english MT
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time
from dataloaders.translation import loader
import torch

from torch import optim
from torch.nn import CrossEntropyLoss
from lstm_attention import EncoderRNN, AttnDecoderRNN, deploy, train_batch

de, en, train_loader, val_loader = loader(batch_size=32)

def sampler(x, pad_idx=1):
    def _skip_eos(row):
        s = []
        for i in row:
            if i == pad_idx:
                break
            s.append(en.vocab.itos[i])
        return s
    x_np = x.data.cpu().numpy().T
    return [' '.join(_skip_eos(row)) for row in x_np]


d_enc_input = 300
d_enc = 256
d_dec_input = 300
d_dec = 256

encoder = EncoderRNN(
    d_enc_input,
    d_enc,
    use_embedding=True,
    vocab_size=len(de.vocab),
    pad_idx=de.vocab.stoi['<pad>'],
)

decoder = AttnDecoderRNN(
    d_dec_input,
    d_enc*2,
    d_dec,
    vocab_size=len(en.vocab),
    pad_idx=en.vocab.stoi['<pad>'],
    bos_token=en.vocab.stoi['<bos>'],
    eos_token=en.vocab.stoi['<eos>'],
)
criterion = CrossEntropyLoss()

if torch.cuda.is_available():
    print("Using cuda")
    encoder.cuda()
    decoder.cuda()
    criterion.cuda()

learning_rate = 0.01
encoder_optimizer = optim.RMSprop(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.RMSprop(decoder.parameters(), lr=learning_rate)

for epoch in range(1, 50):
    for b, batch in enumerate(train_loader):
        if b % 1000 == 0:
            for val_b, val_batch in enumerate(val_loader):
                sampled_outs_ = deploy(encoder, decoder, val_batch.src)
                sampled_outs = sampler(sampled_outs_)

                targets = sampler(val_batch.trg)
                for i in range(min(10, val_batch.src.size(1))):
                    print("----")
                    print("Pred: {}".format(sampled_outs[i]))
                    print("Target: {}".format(targets[i]))
                    print("----", flush=True)
                break

        start = time.time()
        loss = train_batch(encoder, decoder, [encoder_optimizer, decoder_optimizer], criterion,
                           batch.src, batch.trg)
        dur = time.time() - start
        if b % 200 == 0:
            print("e{:2d}b{:3d} Loss is {}, ({:.3f} sec/batch)".format(epoch, b, loss.data[0], dur))
