import torch
import torch.nn.functional as F
import torch.nn as nn

def scaled_dot_product_attention(Q, K, V, mask=None):
    """Compute the scaled dot-product attention.

    Args:
        Q (torch.Tensor): Query tensor of shape (..., seq_len_q, d_k).
        K (torch.Tensor): Key tensor of shape (..., seq_len_k, d_k).
        V (torch.Tensor): Value tensor of shape (..., seq_len_v, d_v), where seq_len_k == seq_len_v.
        mask (torch.Tensor, optional): Masking tensor broadcastable to (..., seq_len_q, seq_len_k). 
            Positions with 0 are masked out (set to -inf in scores). Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - output (torch.Tensor): Attention output of shape (..., seq_len_q, d_v).
            - attention_weights (torch.Tensor): Attention weights of shape (..., seq_len_q, seq_len_k).
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attention_weights = F.softmax(scores, dim=-1)
    
    output = torch.matmul(attention_weights, V)
    return output, attention_weights


class SelfAttention(nn.Module):
    """Self-Attention module applying scaled dot-product attention.

    Attributes:
        embed_size (int): Dimension of embedding vectors.
        query (nn.Linear): Linear layer to project input to query space.
        key (nn.Linear): Linear layer to project input to key space.
        value (nn.Linear): Linear layer to project input to value space.
    """

    def __init__(self, embed_size):
        """Initializes the SelfAttention module.

        Args:
            embed_size (int): Size of input and output embedding dimensions.
        """
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask=None):
        """Forward pass of the Self-Attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_size).
            mask (torch.Tensor, optional): Attention mask of shape (batch_size, 1, seq_len) 
                or broadcastable shape. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_size) after attention.
        """
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)        
        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        return out
