"""Start the E5 retrieval service used by SearchQA."""
import argparse
import os
from pathlib import Path
import sys

from run_experiment import ROOT, load, resource_roots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    roots = resource_roots()
    config = load("retrieval.yaml")["retrieval"]
    paths = {"index_path": roots["data"] / config["index"],
             "corpus_path": roots["data"] / config["corpus"],
             "retriever_model": roots["models"] / config["encoder"]}
    for key in ("index_path", "corpus_path"):
        if not paths[key].is_file():
            raise FileNotFoundError(paths[key])
    if not (paths["retriever_model"] / "config.json").is_file():
        raise FileNotFoundError(paths["retriever_model"])
    command = [sys.executable, str(ROOT / "search_r1/retrieval/retrieval_server.py")]
    for key, path in paths.items():
        command.extend(["--" + key, str(path)])
    command += ["--retriever_name", "e5", "--topk", str(config["topk"]),
                "--port", str(args.port or config["port"])]
    if config["faiss_gpu"]:
        command.append("--faiss_gpu")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(config["gpu"] if args.gpu is None else args.gpu)
    for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        environment[key] = "1"
    os.execve(sys.executable, command, environment)


if __name__ == "__main__":
    main()
