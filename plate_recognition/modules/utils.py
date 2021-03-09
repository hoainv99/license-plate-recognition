from .model import ModelOCR
import torch
import numpy as np
class Vocab():
    def __init__(self, chars):
        self.pad = 0
        self.go = 1
        self.eos = 2
        self.mask_token = 3

        self.chars = chars

        self.c2i = {c:i+4 for i, c in enumerate(chars)}

        self.i2c = {i+4:c for i, c in enumerate(chars)}
        
        self.i2c[0] = '<pad>'
        self.i2c[1] = '<sos>'
        self.i2c[2] = '<eos>'
        self.i2c[3] = '*'

    def encode(self, chars):
        return [self.go] + [self.c2i[c] for c in chars] + [self.eos]
    
    def decode(self, ids):
        first = 1 if self.go in ids else 0
        last = ids.index(self.eos) if self.eos in ids else None
        sent = ''.join([self.i2c[i] for i in ids[first:last]])
        return sent
    
    def __len__(self):
        return len(self.c2i) + 4
    
    def batch_decode(self, arr):
        texts = [self.decode(ids) for ids in arr]
        return texts

    def __str__(self):
        return self.chars
def translate(img, model, max_seq_length=128, sos_token=1, eos_token=2):
    "data: BxCXHxW"
    model.eval()
    device = img.device

    with torch.no_grad():

        src = model.cnn(img)
        memory = model.transformer.forward_encoder(src)


        translated_sentence = [[sos_token]*len(img)]
        max_length = 0

        while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T==eos_token, axis=1)):

            tgt_inp = torch.LongTensor(translated_sentence).to(device)
            
#            output = model(img, tgt_inp, tgt_key_padding_mask=None)
#            output = model.transformer(src, tgt_inp, tgt_key_padding_mask=None)
            output, memory = model.transformer.forward_decoder(tgt_inp, memory)
            output = output.to('cpu')

            values, indices  = torch.topk(output, 2)

            indices = indices[:, -1, 0]
            indices = indices.tolist()

            translated_sentence.append(indices)   
            max_length += 1

            del output 


        translated_sentence = np.asarray(translated_sentence).T

    return translated_sentence
# def translate(img, model, max_seq_length=128, sos_token=1, eos_token=2):
#     "data: BxCXHxW"
#     model.eval()
#     device = img.device

#     with torch.no_grad():

#         src = model.cnn(img)
#         memory = model.seq2seq.forward_encoder(src)

#         translated_sentence = [[sos_token]*len(img)]
#         max_length = 0

#         while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T==eos_token, axis=1)):

#             tgt_inp = torch.LongTensor(translated_sentence).to(device)
            
# #            output = model(img, tgt_inp, tgt_key_padding_mask=None)
# #            output = model.transformer(src, tgt_inp, tgt_key_padding_mask=None)
#             output, memory = model.seq2seq.forward_decoder(tgt_inp, memory)
#             output = output.to('cpu')

#             values, indices  = torch.topk(output, 2)

#             indices = indices[:, -1, 0]
#             indices = indices.tolist()

#             translated_sentence.append(indices)   
#             max_length += 1

#             del output 


#         translated_sentence = np.asarray(translated_sentence).T

#     return translated_sentence
def build_model(vocab):
    vocab = Vocab(vocab)
    device = torch.device('cpu')
    
    model = ModelOCR(len(vocab))
    
    model = model.to(device)

    return model, vocab