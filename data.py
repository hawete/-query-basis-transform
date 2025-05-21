import torch
from torch.utils.data import Dataset

class DummyRetrievalDataset(Dataset):
    def __init__(self, num_samples=1000, embed_dim=128, num_docs=10):
        self.num_samples = num_samples
        self.embed_dim = embed_dim
        self.num_docs = num_docs

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        query_emb = torch.randn(self.embed_dim)
        doc_embs = torch.randn(self.num_docs, self.embed_dim)
        labels = torch.randint(0, 2, (self.num_docs,))  # binary relevance labels

        return query_emb, doc_embs, labels
