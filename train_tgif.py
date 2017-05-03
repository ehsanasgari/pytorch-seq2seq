from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time
from dataloaders.tgif import loader
import torch

from torch import optim
from torch.nn import CrossEntropyLoss
from lstm_attention import EncoderRNN, AttnDecoderRNN, deploy_vid, train_batch_vid
from torch.nn.utils.rnn import pad_packed_sequence

vocab, train_loader, test_loader = loader(batch_size=32)

d_enc_input = 300
d_enc = 256
d_dec_input = 300
d_dec = 128

encoder = EncoderRNN(d_enc_input, d_enc, use_cnn=True, vocab_size=len(vocab))
decoder = AttnDecoderRNN(d_dec_input, d_enc*2, d_dec, vocab_size=len(vocab))
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
    for b, (train_x, train_x_lens, train_y_in, train_y_out) in enumerate(train_loader):
        if b % 100 == 1:
            for test_b, (test_x, test_x_lens, _, test_y) in enumerate(test_loader):
                sampled_outs_ = deploy_vid(test_x, test_x_lens, encoder, decoder)
                sampled_outs = vocab.reverse_apply_list(sampled_outs_.data.cpu().numpy().T)
                targets = vocab.reverse_apply_list(pad_packed_sequence(test_y)[0].data.cpu().numpy().T)

                if test_b == 0:
                    for i in range(10):
                        print("----")
                        print("Pred: {}".format(sampled_outs[i]))
                        print("Target: {}".format(targets[i]))

        start = time.time()
        loss = train_batch_vid(train_x, train_x_lens, train_y_in, train_y_out, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        dur = time.time() - start
        if b % 50 == 0:
            print("e{:2d}b{:3d} Loss is {}, ({:.3f} sec/batch)".format(epoch, b, loss.data[0], dur))