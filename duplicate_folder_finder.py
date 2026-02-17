#!/usr/bin/env python3
"""
Fast duplicate folder and partial duplicate subfolder finder for macOS photo archives.

Design goals:
- Metadata-only scanning by default (no file content reads)
- Read-only on scanned volumes
- Deterministic CSV outputs
- Graceful timeout handling
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

SKETCH_BITS = 1024
SKETCH_MASK = SKETCH_BITS - 1


@dataclass
class FolderNode:
    node_id: int
    volume_name: str
    volume_root: str
    abs_path: str
    rel_path: str
    basename: str
    parent_id: Optional[int]
    depth: int
    child_ids: List[int] = field(default_factory=list)

    direct_size_bytes: int = 0
    direct_file_count: int = 0
    direct_counter: Optional[Counter] = None
    direct_sketch: int = 0

    total_size_bytes: int = 0
    file_count: int = 0
    recursive_counter: Optional[Counter] = None
    recursive_sketch: int = 0


@dataclass
class ScanResult:
    direct_size_bytes: int
    direct_file_count: int
    direct_counter: Optional[Counter]
    direct_sketch: int
    subdirs: List[Tuple[str, str]]  # (name, abs_path)
    errors: List[Tuple[str, str]]


def key_hash(file_name: str, size_bytes: int) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(file_name.encode("utf-8", errors="ignore"))
    h.update(b"\0")
    h.update(str(size_bytes).encode("ascii"))
    return int.from_bytes(h.digest(), "big")


def key_to_sketch_bit(file_name: str, size_bytes: int) -> int:
    return 1 << (key_hash(file_name, size_bytes) & SKETCH_MASK)


def should_skip_name(name: str, include_hidden: bool, service_dirs: set[str]) -> bool:
    if not include_hidden and name.startswith("."):
        return True
    if name in service_dirs:
        return True
    return False


def scan_directory(
    path: str,
    include_hidden: bool,
    service_dirs: set[str],
    max_set_files: int,
) -> ScanResult:
    direct_size = 0
    direct_count = 0
    counter: Counter = Counter()
    use_counter = True
    sketch = 0
    subdirs: List[Tuple[str, str]] = []
    errors: List[Tuple[str, str]] = []

    try:
        with os.scandir(path) as it:
            for entry in it:
                name = entry.name
                if should_skip_name(name, include_hidden, service_dirs):
                    continue
                try:
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        subdirs.append((name, entry.path))
                        continue
                    if entry.is_file(follow_symlinks=False):
                        try:
                            st = entry.stat(follow_symlinks=False)
                        except OSError as exc:
                            errors.append((entry.path, f"stat_error: {exc}"))
                            continue
                        size = int(st.st_size)
                        direct_size += size
                        direct_count += 1
                        sketch |= key_to_sketch_bit(name, size)
                        if use_counter:
                            counter[(name, size)] += 1
                            if direct_count > max_set_files:
                                use_counter = False
                                counter = Counter()
                except OSError as exc:
                    errors.append((entry.path, f"entry_error: {exc}"))
    except OSError as exc:
        errors.append((path, f"scandir_error: {exc}"))

    return ScanResult(
        direct_size_bytes=direct_size,
        direct_file_count=direct_count,
        direct_counter=counter if use_counter else None,
        direct_sketch=sketch,
        subdirs=subdirs,
        errors=errors,
    )


def compute_rel_path(abs_path: str, volume_root: str) -> str:
    try:
        rel = os.path.relpath(abs_path, volume_root)
    except ValueError:
        rel = abs_path
    return "." if rel == "." else rel


def estimate_intersection_with_sketch(sub_sketch: int, parent_sketch: int, sub_count: int, sub_bytes: int) -> Tuple[int, int]:
    if sub_count <= 0:
        return 0, 0
    overlap_bits = (sub_sketch & parent_sketch).bit_count()
    sub_bits = max(1, sub_sketch.bit_count())
    ratio = min(1.0, overlap_bits / sub_bits)
    est_files = int(round(sub_count * ratio))
    est_bytes = int(round(sub_bytes * ratio))
    return est_files, est_bytes


def iter_ancestors(nodes: Dict[int, FolderNode], node: FolderNode, max_depth_up: int) -> Iterable[FolderNode]:
    cur_id = node.parent_id
    hops = 0
    while cur_id is not None and hops < max_depth_up:
        anc = nodes[cur_id]
        yield anc
        cur_id = anc.parent_id
        hops += 1


def compute_partial_match(sub: FolderNode, parent: FolderNode, exclude_sub_from_parent: bool) -> Tuple[int, int, str]:
    """Compute matched files/bytes by (file_name,size), ignoring timestamps and metadata."""
    if sub.recursive_counter is not None and parent.recursive_counter is not None:
        matched_files = 0
        matched_bytes = 0
        parent_counter = parent.recursive_counter - sub.recursive_counter if exclude_sub_from_parent else parent.recursive_counter
        for key, sub_cnt in sub.recursive_counter.items():
            parent_cnt = parent_counter.get(key, 0)
            if parent_cnt <= 0:
                continue
            take = min(sub_cnt, parent_cnt)
            matched_files += take
            matched_bytes += take * key[1]
        return matched_files, matched_bytes, "high"

    confidence = "medium" if (sub.recursive_counter is not None or parent.recursive_counter is not None) else "low"
    est_files, est_bytes = estimate_intersection_with_sketch(
        sub.recursive_sketch,
        parent.recursive_sketch,
        sub.file_count,
        sub.total_size_bytes,
    )
    return est_files, est_bytes, confidence


def write_csv(path: Path, rows: List[dict], fieldnames: List[str], overwrite: bool) -> None:
    mode = "w" if overwrite else "x"
    with path.open(mode=mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast duplicate folder + partial duplicate subfolder finder")
    parser.add_argument("--volumes", nargs="+", required=True, help="Mounted volume paths")
    parser.add_argument("--out", required=True, help="Output directory for CSV files")
    parser.add_argument("--include-hidden", action="store_true", default=False)
    parser.add_argument("--service-dirs", default=".Spotlight-V100,.fseventsd,.Trashes")
    parser.add_argument("--time-budget-seconds", type=int, default=300)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--same-volume-duplicates", dest="same_volume_duplicates", action="store_true", default=True,
                        help="Allow DUPLICATE_FOLDER_FAST groups within the same volume")
    parser.add_argument("--cross-volume-only", dest="same_volume_duplicates", action="store_false",
                        help="Only report DUPLICATE_FOLDER_FAST when matches span volumes")

    parser.add_argument("--prune-duplicate-subtrees", dest="prune_duplicate_subtrees", action="store_true", default=True)
    parser.add_argument("--no-prune-duplicate-subtrees", dest="prune_duplicate_subtrees", action="store_false")

    parser.add_argument("--partial-files-threshold", type=float, default=0.7)
    parser.add_argument("--partial-bytes-threshold", type=float, default=0.7)
    parser.add_argument("--partial-min-matched-files", type=int, default=20)
    parser.add_argument("--partial-max-ancestor-depth", type=int, default=3)

    parser.add_argument("--max-set-files", type=int, default=50000, help="If folder recursive file count exceeds this, use sketch only")
    args = parser.parse_args()

    t0 = time.monotonic()
    deadline = t0 + max(1, args.time_budget_seconds)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    service_dirs = {x.strip() for x in args.service_dirs.split(",") if x.strip()}

    nodes: Dict[int, FolderNode] = {}
    next_node_id = 1
    errors: List[Tuple[str, str]] = []

    pending: Dict = {}
    lock = threading.Lock()

    def add_node(volume_name: str, volume_root: str, abs_path: str, parent_id: Optional[int], depth: int) -> int:
        nonlocal next_node_id
        node_id = next_node_id
        next_node_id += 1
        rel = compute_rel_path(abs_path, volume_root)
        node = FolderNode(
            node_id=node_id,
            volume_name=volume_name,
            volume_root=volume_root,
            abs_path=abs_path,
            rel_path=rel,
            basename=os.path.basename(abs_path.rstrip(os.sep)) or abs_path,
            parent_id=parent_id,
            depth=depth,
        )
        nodes[node_id] = node
        if parent_id is not None:
            nodes[parent_id].child_ids.append(node_id)
        return node_id

    print("[PHASE A] Fast metadata index")
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
        root_ids: List[int] = []
        for vol in args.volumes:
            abs_vol = os.path.abspath(vol)
            vol_name = os.path.basename(abs_vol.rstrip(os.sep)) or abs_vol
            if not os.path.isdir(abs_vol):
                errors.append((abs_vol, "volume_not_directory"))
                continue
            rid = add_node(vol_name, abs_vol, abs_vol, None, 0)
            root_ids.append(rid)
            fut = pool.submit(scan_directory, abs_vol, args.include_hidden, service_dirs, args.max_set_files)
            pending[fut] = rid

        timed_out = False
        while pending:
            now = time.monotonic()
            if now >= deadline:
                timed_out = True
                break

            timeout = max(0.05, min(1.0, deadline - now))
            done, _ = wait(list(pending.keys()), timeout=timeout, return_when=FIRST_COMPLETED)
            if not done:
                continue

            for fut in done:
                node_id = pending.pop(fut)
                node = nodes[node_id]
                try:
                    res = fut.result()
                except Exception as exc:
                    errors.append((node.abs_path, f"scan_exception: {exc}"))
                    continue

                node.direct_size_bytes = res.direct_size_bytes
                node.direct_file_count = res.direct_file_count
                node.direct_counter = res.direct_counter
                node.direct_sketch = res.direct_sketch
                errors.extend(res.errors)

                if time.monotonic() >= deadline:
                    timed_out = True
                    continue

                for child_name, child_abs in sorted(res.subdirs, key=lambda x: x[0]):
                    if should_skip_name(child_name, args.include_hidden, service_dirs):
                        continue
                    child_id = add_node(node.volume_name, node.volume_root, child_abs, node_id, node.depth + 1)
                    fut2 = pool.submit(scan_directory, child_abs, args.include_hidden, service_dirs, args.max_set_files)
                    pending[fut2] = child_id

        if timed_out:
            for fut in pending:
                fut.cancel()
            errors.append(("<scan>", "time_budget_exceeded_partial_results"))

    # Bottom-up recursive aggregation.
    for node in sorted(nodes.values(), key=lambda n: n.depth, reverse=True):
        total_size = node.direct_size_bytes
        total_count = node.direct_file_count
        rec_sketch = node.direct_sketch

        rec_counter: Optional[Counter]
        if node.direct_counter is not None and node.direct_file_count <= args.max_set_files:
            rec_counter = Counter(node.direct_counter)
        else:
            rec_counter = None

        for cid in node.child_ids:
            child = nodes[cid]
            total_size += child.total_size_bytes
            total_count += child.file_count
            rec_sketch |= child.recursive_sketch

            if rec_counter is not None:
                if child.recursive_counter is None:
                    rec_counter = None
                else:
                    rec_counter += child.recursive_counter
                    if total_count > args.max_set_files:
                        rec_counter = None

        node.total_size_bytes = total_size
        node.file_count = total_count
        node.recursive_counter = rec_counter
        node.recursive_sketch = rec_sketch

    elapsed_a = time.monotonic() - t0
    print(f"[PHASE A] done in {elapsed_a:.2f}s")

    print("[PHASE B] Detection")
    # Duplicate folder groups across volumes by (basename, total_size, file_count).
    grouped: Dict[Tuple[str, int, int], List[FolderNode]] = defaultdict(list)
    for n in nodes.values():
        grouped[(n.basename, n.total_size_bytes, n.file_count)].append(n)

    duplicate_groups: List[List[FolderNode]] = []
    for _, group in grouped.items():
        if len(group) < 2:
            continue
        vols = {g.volume_name for g in group}
        if args.same_volume_duplicates or len(vols) >= 2:
            duplicate_groups.append(sorted(group, key=lambda x: (x.volume_name, x.abs_path)))

    duplicate_groups.sort(key=lambda g: (g[0].basename, g[0].total_size_bytes, g[0].file_count, [x.abs_path for x in g]))

    pruned_ids: set[int] = set()
    if args.prune_duplicate_subtrees:
        dup_root_ids = {n.node_id for grp in duplicate_groups for n in grp}
        for node in nodes.values():
            cur = node.parent_id
            while cur is not None:
                if cur in dup_root_ids:
                    pruned_ids.add(node.node_id)
                    break
                cur = nodes[cur].parent_id

    fast_rows: List[dict] = []
    for i, group in enumerate(duplicate_groups, start=1):
        gid = f"G{i:06d}"
        for n in group:
            fast_rows.append(
                {
                    "group_id": gid,
                    "folder_basename": n.basename,
                    "total_size_bytes": n.total_size_bytes,
                    "file_count": n.file_count,
                    "volume": n.volume_name,
                    "folder_abs_path": n.abs_path,
                    "folder_rel_path": n.rel_path,
                    "confidence": "high",
                }
            )

    fast_rows.sort(key=lambda r: (r["group_id"], r["folder_abs_path"]))

    # Partial duplicate detection (ancestor containment + cross-volume same-basename containment,
    # ignoring mtime and all metadata fields except file_name + size_bytes).
    # In-tree check compares B against ancestors A up to depth N and uses A\B to avoid trivial self-containment.
    # Cross-volume check compares folders with the same basename from different volumes.
    partial_rows: List[dict] = []
    rid = 1

    nodes_by_volume: Dict[str, List[FolderNode]] = defaultdict(list)
    for n in nodes.values():
        nodes_by_volume[n.volume_name].append(n)

    for volume_nodes in nodes_by_volume.values():
        for sub in sorted(volume_nodes, key=lambda x: (x.depth, x.abs_path)):
            if sub.file_count == 0:
                continue

            best: Optional[dict] = None
            for anc in iter_ancestors(nodes, sub, args.partial_max_ancestor_depth):
                if anc.file_count <= 0:
                    continue

                matched_files, matched_bytes, confidence = compute_partial_match(
                    sub=sub,
                    parent=anc,
                    exclude_sub_from_parent=True,
                )

                ratio_files = (matched_files / sub.file_count) if sub.file_count else 0.0
                ratio_bytes = (matched_bytes / sub.total_size_bytes) if sub.total_size_bytes else 0.0

                decision = "PARTIAL_DUPLICATE_SUBFOLDER" if (
                    matched_files >= args.partial_min_matched_files
                    and (ratio_files >= args.partial_files_threshold or ratio_bytes >= args.partial_bytes_threshold)
                ) else "NO"

                cand = {
                    "record_id": f"R{rid:08d}",
                    "volume": sub.volume_name,
                    "parent_folder_abs_path": anc.abs_path,
                    "subfolder_abs_path": sub.abs_path,
                    "parent_total_size_bytes": anc.total_size_bytes,
                    "subfolder_total_size_bytes": sub.total_size_bytes,
                    "parent_file_count": anc.file_count,
                    "subfolder_file_count": sub.file_count,
                    "matched_files_count": matched_files,
                    "matched_bytes": matched_bytes,
                    "match_ratio_by_files": f"{ratio_files:.6f}",
                    "match_ratio_by_bytes": f"{ratio_bytes:.6f}",
                    "decision": decision,
                    "confidence": confidence,
                }

                if best is None:
                    best = cand
                else:
                    prev_score = (float(best["match_ratio_by_files"]), float(best["match_ratio_by_bytes"]), best["matched_files_count"])
                    new_score = (ratio_files, ratio_bytes, matched_files)
                    if new_score > prev_score:
                        best = cand

            if best is not None:
                partial_rows.append(best)
                rid += 1

    # Cross-volume partials for same basename (e.g. /Photos/.../ShootA vs /Backup/Photos/.../ShootA).
    by_basename: Dict[str, List[FolderNode]] = defaultdict(list)
    for node in nodes.values():
        if node.file_count > 0:
            by_basename[node.basename].append(node)

    seen_pairs: set[Tuple[str, str]] = set()
    for basename, candidates in by_basename.items():
        if len(candidates) < 2:
            continue
        candidates.sort(key=lambda n: (n.volume_name, n.abs_path))
        for i in range(len(candidates)):
            for j in range(len(candidates)):
                if i == j:
                    continue
                sub = candidates[i]
                parent = candidates[j]
                if sub.volume_name == parent.volume_name:
                    continue
                if sub.file_count > parent.file_count:
                    continue
                pair_key = (sub.abs_path, parent.abs_path)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                matched_files, matched_bytes, confidence = compute_partial_match(
                    sub=sub,
                    parent=parent,
                    exclude_sub_from_parent=False,
                )

                ratio_files = (matched_files / sub.file_count) if sub.file_count else 0.0
                ratio_bytes = (matched_bytes / sub.total_size_bytes) if sub.total_size_bytes else 0.0
                decision = "PARTIAL_DUPLICATE_SUBFOLDER" if (
                    matched_files >= args.partial_min_matched_files
                    and (ratio_files >= args.partial_files_threshold or ratio_bytes >= args.partial_bytes_threshold)
                ) else "NO"

                partial_rows.append(
                    {
                        "record_id": f"R{rid:08d}",
                        "volume": sub.volume_name,
                        "parent_folder_abs_path": parent.abs_path,
                        "subfolder_abs_path": sub.abs_path,
                        "parent_total_size_bytes": parent.total_size_bytes,
                        "subfolder_total_size_bytes": sub.total_size_bytes,
                        "parent_file_count": parent.file_count,
                        "subfolder_file_count": sub.file_count,
                        "matched_files_count": matched_files,
                        "matched_bytes": matched_bytes,
                        "match_ratio_by_files": f"{ratio_files:.6f}",
                        "match_ratio_by_bytes": f"{ratio_bytes:.6f}",
                        "decision": decision,
                        "confidence": confidence,
                    }
                )
                rid += 1

    partial_rows.sort(key=lambda r: (r["subfolder_abs_path"], r["parent_folder_abs_path"], r["record_id"]))

    error_rows = [{"path": p, "error": e} for p, e in sorted(errors, key=lambda x: (x[0], x[1]))]

    write_csv(
        out_dir / "duplicate_folders_fast.csv",
        fast_rows,
        [
            "group_id",
            "folder_basename",
            "total_size_bytes",
            "file_count",
            "volume",
            "folder_abs_path",
            "folder_rel_path",
            "confidence",
        ],
        args.overwrite,
    )

    write_csv(
        out_dir / "partial_duplicate_subfolders.csv",
        partial_rows,
        [
            "record_id",
            "volume",
            "parent_folder_abs_path",
            "subfolder_abs_path",
            "parent_total_size_bytes",
            "subfolder_total_size_bytes",
            "parent_file_count",
            "subfolder_file_count",
            "matched_files_count",
            "matched_bytes",
            "match_ratio_by_files",
            "match_ratio_by_bytes",
            "decision",
            "confidence",
        ],
        args.overwrite,
    )

    write_csv(
        out_dir / "errors.csv",
        error_rows,
        ["path", "error"],
        args.overwrite,
    )

    total_elapsed = time.monotonic() - t0
    print(f"[SUMMARY] elapsed={total_elapsed:.2f}s")
    print(f"[SUMMARY] folders_scanned={len(nodes)}")
    print(f"[SUMMARY] files_scanned={sum(n.direct_file_count for n in nodes.values())}")
    print(f"[SUMMARY] duplicate_folder_groups={len({r['group_id'] for r in fast_rows})}")
    print(f"[SUMMARY] partial_duplicates_found={sum(1 for r in partial_rows if r['decision'] == 'PARTIAL_DUPLICATE_SUBFOLDER')}")
    print(f"[SUMMARY] errors={len(error_rows)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
