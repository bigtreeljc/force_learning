import unittest
from transformer.SubLayers import MultiHeadAttention
import torch

class test(unittest.TestCase):
  def test_self_attention(self):
    # parameters for machine translation
    n_head = 8
    d_model = 512
    d_k = 64
    d_v = 64
    dropout = 0.1
    batch_size = 2
    vocab_size = 1000
    max_seq_len = 50

    slf_attn = MultiHeadAttention(n_head, d_model, 
      d_k, d_v, dropout=dropout)
    print("self attn created")
    input_tensor = torch.randn(batch_size, max_seq_len, d_model).float().to('cpu')
    attn_mask = torch.zeros(batch_size, max_seq_len, max_seq_len).byte().to('cpu')
    # print(attn_mask.size())
    enc_output, enc_slf_attn = slf_attn(input_tensor, input_tensor, input_tensor, mask=attn_mask)
    print(f"enc output {enc_output.size()}")
    print(f"self_attn {attn_mask.size()}")

if __name__ == "__main__":
  unittest.main()
