"""Download public model weights and the Wikipedia retrieval resources."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import tarfile


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "qwen2.5_3b": ("Qwen/Qwen2.5-3B-Instruct", "qwen2.5_3b_instruct"),
    "qwen3_1.7b": ("Qwen/Qwen3-1.7B", "qwen3_1.7b"),
    "e5": ("intfloat/e5-base-v2", "e5-base-v2"),
}
DATA_REVISIONS = {
    "PeterJinGo/wiki-18-e5-index": "a4d31160a035f30764604f4827cd8f1d0315eb86",
    "PeterJinGo/wiki-18-corpus": "69c1c00ffe7c5554c68d8548355cb22e46aabc51",
}
def dataset_file(repository: str, filename: str, directory: Path) -> Path:
    from huggingface_hub import hf_hub_download

    directory.mkdir(parents=True, exist_ok=True)
    existing = directory / filename
    if existing.is_file() and existing.stat().st_size:
        return existing
    return Path(hf_hub_download(
        repo_id=repository, filename=filename, repo_type="dataset",
        revision=DATA_REVISIONS[repository], local_dir=directory,
    ))


def copy_jsonl(source, destination):
    """Check the first record before streaming the unmodified corpus bytes."""
    first = source.readline()
    if not first:
        raise ValueError("The corpus is empty")
    record = json.loads(first)
    if not isinstance(record, dict) or "contents" not in record:
        raise ValueError("The corpus must contain JSON objects with a contents field")
    destination.write(first)
    shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)


def unpack_corpus(archive: Path, output: Path) -> Path:
    """Read a gzip JSONL or a tar stream without extracting archive paths."""
    if output.is_file() and output.stat().st_size:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")
    try:
        if tarfile.is_tarfile(archive):
            found = False
            with tarfile.open(archive, mode="r|*") as bundle:
                for member in bundle:
                    if not member.isfile() or Path(member.name).name != "wiki-18.jsonl":
                        continue
                    if found:
                        raise ValueError("Corpus archive has multiple wiki-18.jsonl members")
                    source = bundle.extractfile(member)
                    if source is None:
                        raise ValueError("Cannot read the corpus member")
                    with source, temporary.open("wb") as destination:
                        copy_jsonl(source, destination)
                    found = True
            if not found:
                raise ValueError("Corpus archive lacks a regular wiki-18.jsonl member")
        else:
            with gzip.open(archive, "rb") as source, temporary.open("wb") as destination:
                copy_jsonl(source, destination)
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output


def download_retrieval(data_root: Path):
    directory = data_root / "searchqa"
    directory.mkdir(parents=True, exist_ok=True)
    index = directory / "e5_Flat.index"
    if not index.is_file() or not index.stat().st_size:
        parts = [dataset_file("PeterJinGo/wiki-18-e5-index", name, directory / "raw_index") for name in ("part_aa", "part_ab")]
        temporary = index.with_name(index.name + ".part")
        try:
            with temporary.open("wb") as destination:
                for part in parts:
                    with part.open("rb") as source:
                        shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
            if temporary.stat().st_size != sum(part.stat().st_size for part in parts):
                raise IOError("Concatenated FAISS index has the wrong size")
            temporary.replace(index)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    corpus = directory / "wiki-18.jsonl"
    if not corpus.is_file() or not corpus.stat().st_size:
        archive = dataset_file("PeterJinGo/wiki-18-corpus", "wiki-18.jsonl.gz", directory / "raw_corpus")
        unpack_corpus(archive, corpus)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=("all", *MODELS), default=[])
    parser.add_argument("--retrieval", action="store_true", help="Download and assemble the Wikipedia corpus and large FAISS index.")
    args = parser.parse_args()
    if not (args.models or args.retrieval):
        parser.error("Select --models or --retrieval")
    data_root = Path(os.environ.get("SDRD_DATA_DIR", ROOT / "data")).expanduser().resolve()
    model_root = Path(os.environ.get("SDRD_MODEL_DIR", ROOT / "models")).expanduser().resolve()
    if args.models:
        from huggingface_hub import snapshot_download

        for name in (MODELS if "all" in args.models else dict.fromkeys(args.models)):
            repository, folder = MODELS[name]
            snapshot_download(repo_id=repository, local_dir=model_root / folder)
    if args.retrieval:
        download_retrieval(data_root)
    print("Requested resources are ready. The complete training and test splits are included in data/searchqa/processed/.")


if __name__ == "__main__":
    main()
