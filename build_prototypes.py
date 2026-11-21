"""Build semantic K-means prototypes from a client's entity descriptions.

Example: python build_prototypes.py --client 0 --clusters 5
Requires numpy and scikit-learn. Uses ./my_config.py.
This baseline uses text embeddings only, not graph structure or GNN training.
"""

import argparse
import importlib.util
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent


def entity_text(entity):
    """Keep each distinct local description once, in source order."""
    name = entity["name"].strip()
    if not name:
        raise ValueError("Entity name must not be empty")
    descriptions = list(dict.fromkeys(
        source["description"].strip() for source in entity["sources"]
        if source.get("description", "").strip()
    ))
    return name + ("\n" + "\n".join(descriptions) if descriptions else "")


def embed_batch(texts, config):
    request = Request(
        config.EMB_BASE_URL.rstrip("/") + "/embeddings",
        data=json.dumps({
            "model": config.EMB_MODEL, "input": texts, "encoding_format": "float",
            "dimensions": config.EMB_DIM,
        }).encode("utf-8"),
        headers={"Authorization": f"Bearer {config.EMB_API_KEY}",
                 "Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=180) as response:
            result = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"Embedding request failed: HTTP {exc.code}") from None
    except URLError:
        raise RuntimeError("Embedding connection failed; check network and base URL") from None
    # The API's index is the position within this batch, not the global entity ID.
    rows = sorted(result["data"], key=lambda row: row["index"])
    if [row["index"] for row in rows] != list(range(len(texts))):
        raise ValueError("Embedding response has missing or duplicate indices")
    vectors = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
    if vectors.shape != (len(texts), config.EMB_DIM):
        raise ValueError(f"Unexpected embedding shape: {vectors.shape}")
    normalize(vectors)  # Validate before proceeding to another paid request.
    return vectors


def normalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or not vectors.shape[0] or not vectors.shape[1]:
        raise ValueError("Expected a nonempty 2D embedding matrix")
    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain NaN or infinity")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("Embeddings have invalid or zero norms")
    return vectors / norms


def cluster_embeddings(vectors, requested_clusters, seed):
    if requested_clusters < 1:
        raise ValueError("Cluster count must be positive")
    normalized = normalize(vectors)
    # Identical vectors cannot form distinct clusters.
    count = min(requested_clusters, len(np.unique(normalized, axis=0)))
    model = KMeans(n_clusters=count, random_state=seed, n_init=10)
    labels = model.fit_predict(normalized)
    # Compact labels even if numerical degeneracy yields fewer occupied clusters.
    occupied, labels = np.unique(labels, return_inverse=True)
    centers = np.stack([normalized[labels == i].mean(axis=0)
                        for i in range(len(occupied))])
    return normalized, centers, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", type=int, default=0)
    parser.add_argument("--clusters", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.client < 0 or args.clusters < 1 or args.batch_size < 1:
        parser.error("client must be nonnegative; clusters and batch-size must be positive")
    graph_path = ROOT / "graphs" / f"client_{args.client}.json"
    output_dir = ROOT / "prototypes"
    array_path = output_dir / f"client_{args.client}.npz"
    metadata_path = output_dir / f"client_{args.client}.json"
    if not args.overwrite and (array_path.exists() or metadata_path.exists()):
        parser.error("Output already exists; use --overwrite to replace it")

    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    if graph["client_id"] != f"client_{args.client}":
        parser.error("Graph client_id does not match requested client")
    entities = graph["entities"]
    if not entities:
        parser.error("Graph contains no entities")
    names = [entity["name"] for entity in entities]
    if len(set(names)) != len(names):
        parser.error("Graph contains duplicate entity names")
    texts = [entity_text(entity) for entity in entities]

    spec = importlib.util.spec_from_file_location(
        "fedgrag_local_config", ROOT / "my_config.py"
    )
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    if not all((config.EMB_BASE_URL, config.EMB_API_KEY, config.EMB_MODEL)) or config.EMB_DIM < 1:
        parser.error("Embedding configuration is incomplete")

    batches = []
    for start in range(0, len(texts), args.batch_size):
        batch = texts[start:start + args.batch_size]
        print(f"Embedding entities {start + 1}-{start + len(batch)}/{len(texts)}", flush=True)
        batches.append(embed_batch(batch, config))
    vectors = np.concatenate(batches)
    normalized, centers, labels = cluster_embeddings(vectors, args.clusters, args.seed)
    clusters = []
    for index, center in enumerate(centers):
        members = np.flatnonzero(labels == index)
        # Representatives are nearest members to the arithmetic centroid.
        order = np.argsort(np.linalg.norm(normalized[members] - center, axis=1), kind="stable")
        clusters.append({
            "prototype_id": f"client_{args.client}_cluster_{index}",
            "vector_index": index, "size": len(members),
            "member_indices": members.tolist(),
            "members": [names[i] for i in members],
            "representative_entities": [names[i] for i in members[order[:5]]],
        })
    metadata = {
        "client_id": graph["client_id"], "method": "semantic_kmeans",
        "graph_file": str(graph_path.relative_to(ROOT)),
        "embedding_model": config.EMB_MODEL, "embedding_base_url": config.EMB_BASE_URL,
        "embedding_dimension": config.EMB_DIM,
        "requested_clusters": args.clusters, "actual_clusters": len(centers),
        "seed": args.seed, "entity_count": len(entities),
        "entity_names": names, "entity_texts": texts,
        "arrays": {
            "entity_embeddings": "Raw API vectors, rows follow entity_names",
            "prototypes": "Arithmetic means of L2-normalized member embeddings; not unit vectors",
            "labels": "Prototype row index for each entity",
        },
        "clusters": clusters,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(array_path, entity_embeddings=vectors, prototypes=centers, labels=labels)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(entities)} entity vectors and {len(centers)} prototypes")
    print(array_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
