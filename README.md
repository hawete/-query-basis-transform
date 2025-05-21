This repository contains the official implementation of our CIKM paper:
“Supervised Learning of Query-Specific Basis Transformations for Dense Retrieval.”

We propose a supervised, model-agnostic method that learns a query-specific transformation matrix M_q , which dynamically adapts document embeddings into query-aligned subspaces. Our approach improves retrieval performance across benchmarks including MS MARCO , TREC DL , and BEIR .
The method supports:

Supervised transformation matrix learning

Neural parameterization (MLP-based)

Integration with ColBERT and SPLADE

Feedback-aware retrieval (manual or pseudo)
