"""Route a question by maximum cosine similarity to client prototypes.

Example: python route_query.py --question "Where was Alex born?" --top-b 2
Uses ./my_config.py. Prints results only; does not save the question,
its vector, or routing results.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from build_prototypes import embed_batch, normalize

ROOT = Path(__file__).resolve().parent


def load_prototypes(directory, config):
    arrays = {path.stem: path for path in directory.glob("client_*.npz")}
    metadata_files = {path.stem: path for path in directory.glob("client_*.json")}
    if not arrays and not metadata_files:
        raise ValueError(f"No client prototypes found in {directory}; run build_prototypes.py first")
    if arrays.keys() != metadata_files.keys():
        raise ValueError("Each client must have both .npz and .json prototype files")
    clients = []
    for client_id in sorted(arrays):
        metadata = json.loads(metadata_files[client_id].read_text(encoding="utf-8"))
        if metadata["client_id"] != client_id:
            raise ValueError(f"Client ID mismatch: {client_id}")
        if (metadata["embedding_model"] != config.EMB_MODEL
                or metadata["embedding_dimension"] != config.EMB_DIM
                or metadata["embedding_base_url"].rstrip("/") != config.EMB_BASE_URL.rstrip("/")):
            raise ValueError(f"Embedding configuration differs from saved prototypes: {client_id}")
        with np.load(arrays[client_id], allow_pickle=False) as data:
            prototypes = normalize(data["prototypes"])
        if prototypes.shape[1] != config.EMB_DIM:
            raise ValueError(f"Wrong prototype dimension: {client_id}")
        clusters = sorted(metadata["clusters"], key=lambda cluster: cluster["vector_index"])
        if (metadata["actual_clusters"] != len(prototypes)
                or [cluster["vector_index"] for cluster in clusters] != list(range(len(prototypes)))):
            raise ValueError(f"Prototype metadata does not match vector rows: {client_id}")
        clients.append({"client_id": client_id, "prototypes": prototypes, "clusters": clusters})
    return clients


def embed_query(question, config):
    return embed_batch([question], config)[0]


def rank_clients(query_vector, clients):
    query = np.asarray(query_vector, dtype=np.float32)
    if query.ndim != 1:
        raise ValueError("Expected a one-dimensional query vector")
    query = normalize(query[None, :])[0]
    ranking = []
    for client in clients:
        prototypes = normalize(client["prototypes"])
        if prototypes.shape[1] != len(query):
            raise ValueError("Query and prototype dimensions differ")
        scores = np.clip(prototypes @ query, -1.0, 1.0)
        best = int(np.argmax(scores))
        cluster = client["clusters"][best]
        ranking.append({
            "client_id": client["client_id"], "score": float(scores[best]),
            "best_prototype_id": cluster["prototype_id"],
            "representative_entities": cluster.get("representative_entities", []),
        })
    return sorted(ranking, key=lambda item: (-item["score"], item["client_id"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-b", type=int, default=2)
    args = parser.parse_args()
    question = args.question.strip()
    if not question or args.top_b < 1:
        parser.error("question must not be empty and top-b must be positive")
    try:
        spec = importlib.util.spec_from_file_location(
            "fedgrag_local_config", ROOT / "my_config.py"
        )
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        if not all((config.EMB_BASE_URL, config.EMB_API_KEY, config.EMB_MODEL)) or config.EMB_DIM < 1:
            raise ValueError("Embedding configuration is incomplete")
        clients = load_prototypes(ROOT / "prototypes", config)
        if args.top_b > len(clients):
            raise ValueError(f"top-b={args.top_b} exceeds the {len(clients)} available clients")
        ranking = rank_clients(embed_query(question, config), clients)
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(f"Loaded {len(clients)} clients")
    print("Selected: " + ", ".join(item["client_id"] for item in ranking[:args.top_b]))
    print("\nRanking (cosine similarity, not probability):")
    for index, item in enumerate(ranking):
        marker = "*" if index < args.top_b else " "
        print(f"{marker} {item['client_id']}  {item['score']:.6f}  best: {item['best_prototype_id']}")
        print("    representatives: " + ", ".join(item["representative_entities"]))


if __name__ == "__main__":
    main()
