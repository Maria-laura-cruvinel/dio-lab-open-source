import os
import re
import json
import argparse
from pathlib import Path

import numpy as np
import faiss
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> list[dict]:
    """Retorna lista de {'text':..., 'page': int} por página."""
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"text": text, "page": i + 1})
    return pages


def clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """
    Chunking simples por caracteres, com overlap.
    Para projetos maiores, você pode trocar por chunking por tokens/sentenças.
    """
    text = clean_text(text)
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def build_corpus(inputs_dir: Path, chunk_size: int, overlap: int) -> list[dict]:
    """
    Retorna lista de chunks com metadados:
    {
      "id": int,
      "text": str,
      "source_file": str,
      "page": int|None
    }
    """
    corpus = []
    idx = 0

    for path in sorted(inputs_dir.glob("*")):
        if path.suffix.lower() == ".txt":
            full = read_txt(path)
            chunks = chunk_text(full, chunk_size, overlap)
            for c in chunks:
                corpus.append({
                    "id": idx,
                    "text": c,
                    "source_file": path.name,
                    "page": None
                })
                idx += 1

        elif path.suffix.lower() == ".pdf":
            pages = read_pdf(path)
            for p in pages:
                chunks = chunk_text(p["text"], chunk_size, overlap)
                for c in chunks:
                    corpus.append({
                        "id": idx,
                        "text": c,
                        "source_file": path.name,
                        "page": p["page"]
                    })
                    idx += 1

    return corpus


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default="inputs", help="Pasta com PDFs/TXTs")
    parser.add_argument("--out", default="data", help="Pasta de saída do índice")
    parser.add_argument("--chunk_size", type=int, default=900)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    inputs_dir = Path(args.inputs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = build_corpus(inputs_dir, args.chunk_size, args.overlap)
    if not corpus:
        raise SystemExit(f"Nenhum .pdf ou .txt encontrado em: {inputs_dir.resolve()}")

    texts = [c["text"] for c in corpus]

    embedder = SentenceTransformer(args.model)
    embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity (com embeddings normalizados)
    index.add(embeddings)

    faiss.write_index(index, str(out_dir / "faiss.index"))

    with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in corpus:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    meta = {
        "embedding_model": args.model,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "num_chunks": len(corpus),
        "dim": dim
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("✅ Indexação concluída!")
    print(f"- Chunks: {len(corpus)}")
    print(f"- Índice: {(out_dir / 'faiss.index').resolve()}")
    print(f"- Metadados: {(out_dir / 'meta.json').resolve()}")


if __name__ == "__main__":
    main()