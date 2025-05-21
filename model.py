import torch
import torch.nn as nn

class QueryBasisTransformModel(nn.Module):
    def __init__(self, embed_dim, hidden_dim=128):
        super(QueryBasisTransformModel, self).__init__()
        # MLP to generate query-specific transformation matrix M_q
        # For simplicity, output a square matrix flattened
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim * embed_dim)
        )

    def forward(self, query_emb, doc_embs):
        """
        query_emb: Tensor [batch_size, embed_dim]
        doc_embs: Tensor [batch_size, num_docs, embed_dim]
        """
        batch_size = query_emb.size(0)
        M_q_flat = self.mlp(query_emb)  # [batch_size, embed_dim*embed_dim]
        M_q = M_q_flat.view(batch_size, self.embed_dim, self.embed_dim)  # [batch_size, embed_dim, embed_dim]

        # Apply query-specific transform to each document embedding
        # doc_embs: [batch_size, num_docs, embed_dim]
        doc_embs_transformed = torch.bmm(doc_embs, M_q)  # batch matrix multiply

        return doc_embs_transformed
