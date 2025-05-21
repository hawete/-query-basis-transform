import torch
from src.model import QueryBasisTransformModel
from src.data import DummyRetrievalDataset
from torch.utils.data import DataLoader

def evaluate(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        for query_emb, doc_embs, labels in dataloader:
            query_emb = query_emb.to(device)
            doc_embs = doc_embs.to(device)
            labels = labels.to(device)

            transformed_docs = model(query_emb, doc_embs)
            # Just print shape for now
            print("Transformed docs shape:", transformed_docs.shape)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_dim = 128

    model = QueryBasisTransformModel(embed_dim).to(device)
    dataset = DummyRetrievalDataset(num_samples=10, embed_dim=embed_dim, num_docs=5)
    dataloader = DataLoader(dataset, batch_size=2)

    evaluate(model, dataloader, device)

if __name__ == "__main__":
    main()
