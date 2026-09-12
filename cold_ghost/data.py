"""Explicit independent training manifests and exhaustive evaluation-image guard."""
from __future__ import annotations
import hashlib
import json
import struct
from pathlib import Path
from PIL import Image, ImageOps
from .config import ROOT
from .checkpoints import file_sha256


def canonical_image_hash(image):
    """Exact EXIF-normalized RGB pixel identity (not a perceptual near-duplicate test)."""
    close = not isinstance(image, Image.Image)
    im = Image.open(image) if close else image
    try:
        rgb = ImageOps.exif_transpose(im).convert("RGB")
        return hashlib.sha256(struct.pack(">II", *rgb.size)+rgb.tobytes()).hexdigest()
    finally:
        if close:
            im.close()


def read_manifest(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Required independent-data manifest missing: {path}; benchmark images are never substituted")
    records, sample_ids = [], set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        required = {"sample_id", "image_id", "image_path", "question"}
        if not required <= record.keys() or any(record[k] is None or not str(record[k]).strip() for k in required) or not isinstance(record.get("question"), str):
            raise ValueError(f"{path}:{line_number}: require nonempty {sorted(required)}")
        record = {key: str(value) for key, value in record.items()}
        if record["sample_id"] in sample_ids:
            raise ValueError(f"Duplicate sample_id: {record['sample_id']}")
        sample_ids.add(record["sample_id"])
        image = Path(record["image_path"]).expanduser()
        # All manifest image paths are relative to the manifest's directory.
        if not image.is_absolute():
            image = path.parent/image
        if not image.is_file():
            raise FileNotFoundError(f"{path}:{line_number}: missing image {image}")
        record["image_path"] = str(image.resolve())
        record["sha256_rgb"] = canonical_image_hash(image)
        records.append(record)
    if not records:
        raise ValueError(f"Empty manifest: {path}")
    return records


def load_eval_index(path, required_tasks=None):
    from .experiments import benchmark_names
    required_tasks = set(benchmark_names() if required_tasks is None else required_tasks)
    path = Path(path)
    meta_path = path.with_suffix(".meta.json")
    if not path.is_file() or not meta_path.is_file():
        raise FileNotFoundError("Create a complete eval-image index first with `python -m cold_ghost.cli index-eval`; both JSONL and .meta.json are required")
    meta = json.loads(meta_path.read_text())
    if not meta.get("complete") or meta.get("index_sha256") != file_sha256(path):
        raise ValueError("Evaluation-image index is partial, modified, or missing its completion checksum")
    if not required_tasks <= set(meta.get("requested_tasks", [])):
        raise ValueError(f"Evaluation-image index missing tasks: {sorted(required_tasks-set(meta.get('requested_tasks', [])))}")
    hashes, ids = set(), set()
    for line in path.read_text().splitlines():
        record = json.loads(line)
        hashes.add(record["sha256_rgb"])
        ids.update(str(x) for x in record.get("image_ids", []))
    if not hashes:
        raise ValueError("Evaluation-image index has no images")
    return hashes, ids, meta


def audit_data(train_path, val_path, eval_index, *, required_tasks=None):
    train, val = read_manifest(train_path), read_manifest(val_path)
    forbidden_hashes, forbidden_ids, meta = load_eval_index(eval_index, required_tasks)
    for image in (ROOT/"bench_data").glob("*.jpg"):
        forbidden_hashes.add(canonical_image_hash(image))
    train_hash = {r["sha256_rgb"] for r in train}
    train_id = {r["image_id"] for r in train}
    if train_hash & {r["sha256_rgb"] for r in val} or train_id & {r["image_id"] for r in val}:
        raise ValueError("Image leakage between Ghost train and validation manifests")
    if {r["sample_id"] for r in train} & {r["sample_id"] for r in val}:
        raise ValueError("sample_id overlap between train and validation")
    for split, rows in (("train", train), ("validation", val)):
        leaks = [r["sample_id"] for r in rows if r["sha256_rgb"] in forbidden_hashes or r["image_id"] in forbidden_ids]
        if leaks:
            raise ValueError(f"{split} contains evaluation/bench images: {leaks[:10]}")
    report = dict(train_samples=len(train), val_samples=len(val), train_images=len(train_hash),
                  val_images=len({r["sha256_rgb"] for r in val}), eval_images=len(forbidden_hashes),
                  train_manifest_sha256=file_sha256(train_path), val_manifest_sha256=file_sha256(val_path),
                  eval_index_sha256=meta["index_sha256"], image_overlap=False)
    return train, val, report


def split_manifest(source, train_path, val_path, val_fraction=0.1):
    """Deterministic image-group split; identical image IDs OR pixels stay together."""
    for destination in (train_path, val_path):
        if Path(destination).exists():
            raise FileExistsError(f"Refusing to overwrite manifest {destination}")
    if Path(train_path).resolve() == Path(val_path).resolve():
        raise ValueError("Train and validation destinations must differ")
    rows = read_manifest(source)
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one")
    parent = list(range(len(rows)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        for key in (("id", row["image_id"]), ("hash", row["sha256_rgb"])):
            if key in seen:
                parent[root(i)] = root(seen[key])
            seen[key] = i
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(root(i), []).append(row)
    if len(groups) < 2:
        raise ValueError("At least two independent image groups are required")
    ordered = sorted(groups.values(), key=lambda g: hashlib.sha256(("42:"+min(r['sha256_rgb'] for r in g)).encode()).hexdigest())
    nval = max(1, min(len(ordered)-1, round(len(ordered)*val_fraction)))
    for destination, selected in ((train_path, ordered[nval:]), (val_path, ordered[:nval])):
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite manifest {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            for group in selected:
                for row in group:
                    stream.write(json.dumps({k: row[k] for k in ("sample_id", "image_id", "image_path", "question")}, ensure_ascii=False)+"\n")
    return dict(train_images=len(ordered)-nval, val_images=nval)


def index_evaluation_images(destination):
    """Enumerate original task evaluation documents, never answers, and hash images.

    Downloads only through the original lmms-eval task definitions. Completion
    is recorded only after ALL requested tasks and their documents succeed.
    """
    from lmms_eval.tasks import TaskManager, get_task_dict
    from .experiments import benchmark_names
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix+".partial")
    requested = benchmark_names()
    manager = TaskManager()
    counts, unique = {}, set()
    def leaves(tree):
        for name, value in tree.items():
            if isinstance(value, dict):
                yield from leaves(value)
            elif hasattr(value, "dataset"):
                yield str(name), value
            elif isinstance(value, tuple) and len(value)>1 and hasattr(value[1], "dataset"):
                yield str(name), value[1]
    with temp.open("w", encoding="utf-8") as stream:
        for requested_name in requested:
            task_tree = get_task_dict([requested_name], task_manager=manager)
            tasks = list(leaves(task_tree))
            if not tasks:
                raise RuntimeError(f"No leaf tasks resolved for {requested_name}")
            count = 0
            for name, task in tasks:
                # Read the image-bearing dataset, not dataset_no_image/eval request IDs.
                split = task.config.test_split if task.has_test_docs() else task.config.validation_split
                if not split or split not in task.dataset:
                    raise RuntimeError(f"Cannot resolve evaluation split for {name}")
                docs = task.dataset[split]
                for doc_idx, doc in enumerate(docs):
                    visuals = task.doc_to_visual(doc)
                    if isinstance(visuals, Image.Image):
                        visuals = [visuals]
                    if not visuals:
                        raise RuntimeError(f"{name}:{doc_idx} has no visual; cannot certify image isolation")
                    image_ids = [str(doc[k]) for k in ("image_id", "image_path", "image_file", "file_name") if k in doc and isinstance(doc[k], (str, int))]
                    for image in visuals:
                        digest = canonical_image_hash(image)
                        key = (digest, tuple(image_ids))
                        if key not in unique:
                            stream.write(json.dumps(dict(sha256_rgb=digest, image_ids=image_ids, task=name), ensure_ascii=False)+"\n")
                            unique.add(key)
                        count += 1
                    if doc_idx % 1000 == 0:
                        print(f"[index] {requested_name}/{name}: {doc_idx+1}/{len(docs)}", flush=True)
                del docs
            if count == 0:
                raise RuntimeError(f"No evaluated images for {requested_name}")
            counts[requested_name] = count
            del task_tree, tasks
    temp.replace(destination)
    metadata = dict(schema=1, complete=True, requested_tasks=requested, image_occurrences=counts,
                    unique_records=len(unique), index_sha256=file_sha256(destination),
                    policy="exact EXIF-normalized RGB hashes plus image IDs; no perceptual near-duplicate guarantee")
    destination.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return metadata
