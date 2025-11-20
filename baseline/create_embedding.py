import argparse
import torch
import numpy as np
from tqdm import tqdm
from langchain.document_loaders import HuggingFaceDatasetLoader
from sentence_transformers import SentenceTransformer


device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    output_file = args.output
    
    dataset_name = "rabuahmad/climatecheck_publications_corpus"
    page_content_column = "abstract"

    loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)
    publications = loader.load()
    print("[INFO] Publications dataset loaded")

    model = SentenceTransformer(args.model, device='cuda')
    model.max_seq_length = 4096
    model.to(device)
    print(f"[INFO] Model loaded {args.model}")

    abstracts = [entry.page_content for entry in publications]
    metadata = [entry.metadata for entry in publications]

    print("[INFO] Start encoding abstracts")
    abstract_embeddings = []
    batch_size = 512
    for i in tqdm(range(0, len(abstracts), batch_size)):
        batch = abstracts[i:i+batch_size]
        batch_emb = model.encode(batch, device='cuda')
        abstract_embeddings.append(batch_emb)

    print("[INFO] Encoding finished")
    abstract_embeddings = np.vstack(abstract_embeddings)

    np.save(output_file, abstract_embeddings)
    print("[INFO] Embeddings saved")