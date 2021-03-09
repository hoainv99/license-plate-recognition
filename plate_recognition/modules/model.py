from torch import nn
from .seq2seq import Seq2Seq
from .cnn import CNN
from .transformer import LanguageTransformer

# class ModelOCR(nn.Module):
#     def __init__(self, vocab_size):
        
#         super(ModelOCR, self).__init__()
#         self.cnn = CNN()
#         self.seq2seq = Seq2Seq(vocab_size)
        
#     def forward(self, img, tgt_input):

#         """
#         Shape:
#             - img: (N, C, H, W)
#             - tgt_input: (T, N)
#             - tgt_key_padding_mask: (N, T)
#             - output: b t v
#         """
        
#         src = self.cnn(img)
#         tgt_input = tgt_input.transpose(1,0)
#         outputs = self.seq2seq(src, tgt_input)

#         return outputs

class ModelOCR(nn.Module):
    def __init__(self, vocab_size):
        
        super(ModelOCR, self).__init__()
        self.cnn = CNN()
        self.transformer = LanguageTransformer(vocab_size)
        
    def forward(self, img, tgt_input, tgt_key_padding_mask):

        """
        Shape:
            - img: (N, C, H, W)
            - tgt_input: (T, N)
            - tgt_key_padding_mask: (N, T)
            - output: b t v
        """
        
        src = self.cnn(img)

        outputs = self.transformer(src, tgt_input, tgt_key_padding_mask=tgt_key_padding_mask)

        return outputs