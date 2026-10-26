# Modified for the anonymous main training and evaluation release.
from typing import List, Optional
import argparse
import os
from pathlib import Path

import faiss
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import datasets

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=4)
    return corpus


def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results


def load_model(model_path: str, use_fp16: bool = False):
    # model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer


def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")


class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        # processing query for different encoders
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        inputs = self.tokenizer(
            query_list, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        output = self.model(**inputs, return_dict=True)
        query_emb = pooling(
            output.pooler_output, output.last_hidden_state, inputs["attention_mask"],
            self.pooling_method,
        )
        query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")

        return query_emb


class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk

        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def _search(self, query: str, num: int, return_score: bool):
        raise NotImplementedError

    def _batch_search(self, query_list: List[str], num: int, return_score: bool):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False):
        return self._search(query, num, return_score)

    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        return self._batch_search(query_list, num, return_score)


class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        self.index = faiss.read_index(self.index_path)
        if config.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)

        self.corpus = load_corpus(self.corpus_path)
        self.encoder = Encoder(
            model_name=self.retrieval_method,
            model_path=config.retrieval_model_path,
            pooling_method=config.retrieval_pooling_method,
            max_length=config.retrieval_query_max_length,
            use_fp16=config.retrieval_use_fp16,
        )
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0]
        scores = scores[0]
        results = load_docs(self.corpus, idxs)
        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk

        results = []
        scores = []
        for start_idx in range(0, len(query_list), self.batch_size):
            query_batch = query_list[start_idx : start_idx + self.batch_size]
            batch_emb = self.encoder.encode(query_batch)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)

            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()
            # load_docs is not vectorized, but is a python list approach
            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            # chunk them back
            batch_results = [batch_results[i * num : (i + 1) * num] for i in range(len(batch_idxs))]
            results.extend(batch_results)
            scores.extend(batch_scores)
        if return_score:
            return results, scores
        else:
            return results


def get_retriever(config):
    if config.retrieval_method != "e5":
        raise ValueError("The main experiment uses the E5 dense retriever")
    return DenseRetriever(config)


class Config:
    """Configuration for the E5 retrieval service."""

    def __init__(
        self,
        retrieval_method: str = "e5",
        retrieval_topk: int = 3,
        index_path: str = "data/searchqa/e5_Flat.index",
        corpus_path: str = "data/searchqa/wiki-18.jsonl",
        dataset_path: str = "./data",
        data_split: str = "train",
        faiss_gpu: bool = True,
        retrieval_model_path: str = "models/e5-base-v2",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128,
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size


class QueryRequest(BaseModel):
    query: str
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()


@app.get("/health")
def health_endpoint():
    return {
        "status": "ready",
        "index_path": str(Path(config.index_path).resolve()),
        "corpus_path": str(Path(config.corpus_path).resolve()),
        "retriever_model": config.retrieval_model_path,
        "topk": config.retrieval_topk,
        "faiss_gpu": config.faiss_gpu,
        "documents": len(retriever.corpus),
    }


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts a single query and performs retrieval.
    Input format:
    {
      "query": "What is Python?",
      "topk": 3,
      "return_scores": true
    }
    """
    if not request.topk:
        request.topk = config.retrieval_topk  # fallback to default

    # Perform retrieval
    if request.return_scores:
        results, scores = retriever.search(query=request.query, num=request.topk, return_score=True)
    else:
        results = retriever.search(query=request.query, num=request.topk, return_score=False)
        scores = None

    # Format response
    resp = []
    if request.return_scores and scores is not None:
        # If scores are returned, combine them with results
        combined = []
        for doc, score in zip(results, scores):
            # Convert numpy float32 to regular Python float for JSON serialization
            combined.append({"document": doc, "score": float(score)})
        resp.append(combined)
    else:
        resp.append(results)
    return {"result": resp}


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[2]
    data_root = Path(os.environ.get("SDRD_DATA_DIR", project_root / "data")).expanduser().resolve()
    model_root = Path(os.environ.get("SDRD_MODEL_DIR", project_root / "models")).expanduser().resolve()
    parser = argparse.ArgumentParser(description="Launch the E5 retrieval service.")
    parser.add_argument(
        "--index_path", type=str,
        default=str(data_root / "searchqa/e5_Flat.index"),
        help="Corpus indexing file."
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default=str(data_root / "searchqa/wiki-18.jsonl"),
        help="Local corpus file.",
    )
    parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")
    parser.add_argument("--retriever_name", choices=("e5",), default="e5")
    parser.add_argument(
        "--retriever_model", type=str, default=str(model_root / "e5-base-v2"), help="Local E5 model directory."
    )
    parser.add_argument("--faiss_gpu", action="store_true", help="Use GPU for computation")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the FastAPI server on.")

    args = parser.parse_args()

    config = Config(
        retrieval_method=args.retriever_name,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=512,
    )

    retriever = get_retriever(config)

    uvicorn.run(app, host="0.0.0.0", port=args.port)
