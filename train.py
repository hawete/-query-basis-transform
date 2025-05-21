import torch
from torch.utils.data import DataLoader
from src.model import QueryBasisTransformModel
from src.data import DummyRetrievalDataset

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for query_emb, doc_embs, labels in dataloader:
        query_emb = query_emb.to(device)
        doc_embs = doc_embs.to(device)
        labels = labels.to(device).float()

        optimizer.zero_grad()
        transformed_docs = model(query_emb, doc_embs)  # [batch, num_docs, embed_dim]

        # Dummy loss: difference between sum of transformed docs and relevance labels
        # Replace with real loss (e.g., triplet loss) later
        scores = transformed_docs.sum(dim=2)  # sum over embed_dim -> [batch, num_docs]
        loss = ((scores - labels) ** 2).mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embed_dim = 128
    model = QueryBasisTransformModel(embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataset = DummyRetrievalDataset(num_samples=100, embed_dim=embed_dim, num_docs=5)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    for epoch in range(3):
        loss = train_one_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

if __name__ == "__main__":
    main()
