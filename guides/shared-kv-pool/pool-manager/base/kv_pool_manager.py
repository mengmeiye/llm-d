#!/usr/bin/env python3
"""Node-local lifecycle manager for shared vLLM KV-cache pools.

A pool is a directory on node memory (tmpfs) that every vLLM replica on the
node mounts as its filesystem KV-offload tier with `managed_pool` set. The
pool must outlive any one replica, so its lifecycle belongs to this manager,
which runs as a DaemonSet:

  create   Publish a pool generation: build it under a temporary name, write
           pool.json, and rename it into place, so no replica ever sees a
           partial pool. Recreating with the same config keeps the published
           generation; a different config is refused.
  destroy  Remove a generation, but only while no replica holds its shared
           lock, and only if its generation matches the one given. The pool
           is marked DESTROYING under the exclusive lock first, so a replica
           that attaches mid-destroy rejects it.
  evict    Keep the pool between watermarks by deleting the least recently
           used block files (vLLM refreshes block mtimes on access), and
           remove temporary files that a crashed writer left behind.
  serve    create, then evict in a loop until SIGTERM. Stopping the manager
           leaves the pool in place. After an explicit destroy the manager
           idles unready instead of recreating the pool; restart it to
           publish a new generation.
  status   Print a pool's metadata, usage, and whether replicas are attached.

Every create/destroy/recovery for a pool name holds that name's management
guard (a flock on <base>/.<name>.guard, never deleted), so management
operations on one name never interleave. Replicas never take the guard.

The pool layout and the attach protocol are the vLLM side's contract; see
vllm/v1/kv_offload/tiering/fs/pool.py.
"""

import argparse
import contextlib
import errno
import fcntl
import json
import logging
import os
import secrets
import shutil
import signal
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass

POOL_MAGIC = "vllm-kv-pool"
POOL_FORMAT_VERSION = 1
METADATA_FILE = "pool.json"
LOCK_FILE = "lock"
DATA_DIR = "data"
STATE_READY = "READY"
STATE_DESTROYING = "DESTROYING"
# vLLM writes a block to "<name>.bin_<random>.tmp" and renames it into place.
TMP_SUFFIX = ".tmp"

log = logging.getLogger("kv-pool-manager")


class PoolError(Exception):
    pass


class PoolBusy(PoolError):
    pass


@contextlib.contextmanager
def management_guard(base: str, name: str) -> Iterator[None]:
    fd = os.open(os.path.join(base, f".{name}.guard"), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def read_metadata(root: str) -> dict:
    with open(os.path.join(root, METADATA_FILE), "rb") as f:
        metadata = json.load(f)
    if metadata.get("magic") != POOL_MAGIC or metadata.get("version") != POOL_FORMAT_VERSION:
        raise PoolError(f"{root} is not a version-{POOL_FORMAT_VERSION} KV pool")
    return metadata


def write_metadata(root: str, metadata: dict) -> None:
    tmp = os.path.join(root, f".{METADATA_FILE}{TMP_SUFFIX}")
    with open(tmp, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, os.path.join(root, METADATA_FILE))


def try_lock(root: str, mode: int) -> int | None:
    """Take a non-blocking flock on the pool lock; None if it is held."""
    fd = os.open(os.path.join(root, LOCK_FILE), os.O_RDWR)
    try:
        fcntl.flock(fd, mode | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
            return None
        raise
    return fd


def remove_tree(path: str) -> None:
    # Rename first so the name is free at once and a partial removal never
    # looks like a pool.
    doomed = f"{path}.removing-{secrets.token_hex(4)}"
    os.rename(path, doomed)
    shutil.rmtree(doomed, ignore_errors=True)


def recover(base: str, name: str) -> None:
    """Clean up after a manager that crashed mid-create or mid-destroy.

    Called with the management guard held.
    """
    for entry in os.listdir(base):
        if entry.startswith(f".{name}.creating-") or entry.startswith(f"{name}.removing-"):
            log.info("removing leftover %s", entry)
            shutil.rmtree(os.path.join(base, entry), ignore_errors=True)
    root = os.path.join(base, name)
    if not os.path.isdir(root):
        return
    try:
        metadata = read_metadata(root)
    except (OSError, ValueError, PoolError):
        log.warning("removing unreadable pool %s", root)
        remove_tree(root)
        return
    if metadata.get("state") == STATE_DESTROYING:
        fd = try_lock(root, fcntl.LOCK_EX)
        if fd is None:
            raise PoolBusy(f"{root} is DESTROYING but still attached")
        try:
            remove_tree(root)
        finally:
            os.close(fd)
        log.info("finished interrupted destroy of %s", root)


def create(base: str, name: str, config: dict) -> dict:
    """Publish a pool generation, or return the matching published one."""
    os.makedirs(base, exist_ok=True)
    with management_guard(base, name):
        recover(base, name)
        root = os.path.join(base, name)
        if os.path.isdir(root):
            metadata = read_metadata(root)
            if metadata.get("config") != config:
                raise PoolError(
                    f"{root} is generation {metadata['generation']} with config "
                    f"{metadata.get('config')}; destroy it before creating it "
                    f"with {config}"
                )
            return metadata
        tmp = os.path.join(base, f".{name}.creating-{secrets.token_hex(4)}")
        os.makedirs(os.path.join(tmp, DATA_DIR))
        open(os.path.join(tmp, LOCK_FILE), "w").close()
        # Model servers run under their own UID: they open the lock read-write
        # and create block files under data/.
        os.chmod(os.path.join(tmp, DATA_DIR), 0o777)
        os.chmod(os.path.join(tmp, LOCK_FILE), 0o666)
        metadata = {
            "magic": POOL_MAGIC,
            "version": POOL_FORMAT_VERSION,
            "generation": secrets.token_hex(16),
            "state": STATE_READY,
            "config": config,
            "createdAt": int(time.time()),
        }
        write_metadata(tmp, metadata)
        os.rename(tmp, root)
        log.info("created %s generation %s", root, metadata["generation"])
        return metadata


def destroy(base: str, name: str, generation: str) -> None:
    with management_guard(base, name):
        recover(base, name)
        root = os.path.join(base, name)
        if not os.path.isdir(root):
            raise PoolError(f"no pool at {root}")
        metadata = read_metadata(root)
        if metadata["generation"] != generation:
            raise PoolError(
                f"{root} is generation {metadata['generation']}, not {generation}; "
                "refusing to destroy a replacement"
            )
        fd = try_lock(root, fcntl.LOCK_EX)
        if fd is None:
            raise PoolBusy(f"{root} is attached by at least one replica")
        try:
            metadata["state"] = STATE_DESTROYING
            write_metadata(root, metadata)
            remove_tree(root)
        finally:
            os.close(fd)
        log.info("destroyed %s generation %s", root, generation)


def attached(root: str) -> bool:
    fd = try_lock(root, fcntl.LOCK_EX)
    if fd is None:
        return True
    os.close(fd)
    return False


@dataclass
class Usage:
    files: int
    bytes: int


def scan(data_dir: str, stale_tmp_s: float) -> Iterator[tuple[float, int, str]]:
    """Yield (mtime, size, path) for block files; remove stale temp files."""
    now = time.time()
    for dirpath, _, filenames in os.walk(data_dir):
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            try:
                st = os.stat(path)
            except FileNotFoundError:
                continue
            if filename.endswith(TMP_SUFFIX):
                if now - st.st_mtime > stale_tmp_s:
                    with contextlib.suppress(FileNotFoundError):
                        os.remove(path)
                continue
            if filename.endswith(".bin"):
                yield st.st_mtime, st.st_blocks * 512, path


def usage(data_dir: str) -> Usage:
    files = total = 0
    for _, size, _ in scan(data_dir, stale_tmp_s=float("inf")):
        files += 1
        total += size
    return Usage(files, total)


def evict_once(
    root: str, capacity: int, high: float, low: float, stale_tmp_s: float
) -> Usage:
    """Delete least recently used blocks until usage is at most `low`."""
    data_dir = os.path.join(root, DATA_DIR)
    blocks = list(scan(data_dir, stale_tmp_s))
    total = sum(size for _, size, _ in blocks)
    if total <= high * capacity:
        return Usage(len(blocks), total)
    target = low * capacity
    freed = removed = 0
    for _, size, path in sorted(blocks):
        if total - freed <= target:
            break
        # A replica may load a block an instant before it goes; its load then
        # fails and it recomputes, which vLLM handles as a miss.
        with contextlib.suppress(FileNotFoundError):
            os.remove(path)
            freed += size
            removed += 1
    log.info(
        "evicted %d blocks (%.1f MiB) from %s; %.1f%% of capacity used",
        removed,
        freed / 2**20,
        root,
        100 * (total - freed) / capacity,
    )
    return Usage(len(blocks) - removed, total - freed)


def capacity_of(root: str, configured: int) -> int:
    if configured > 0:
        return configured
    st = os.statvfs(root)
    return st.f_blocks * st.f_frsize


def status(base: str, name: str) -> dict:
    root = os.path.join(base, name)
    metadata = read_metadata(root)
    u = usage(os.path.join(root, DATA_DIR))
    return {
        "pool": root,
        "generation": metadata["generation"],
        "state": metadata["state"],
        "config": metadata.get("config"),
        "attached": attached(root),
        "blockFiles": u.files,
        "usedBytes": u.bytes,
    }


def serve(args: argparse.Namespace, config: dict) -> None:
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    metadata = create(args.base_dir, args.name, config)
    root = os.path.join(args.base_dir, args.name)
    ready_file = os.path.join(args.base_dir, f".{args.name}.ready")
    with open(ready_file, "w") as f:
        f.write(metadata["generation"])
    print(json.dumps({"pool": root, "generation": metadata["generation"]}), flush=True)
    capacity = capacity_of(root, args.capacity_bytes)
    log.info(
        "managing %s: capacity %.1f GiB, evict above %.0f%% down to %.0f%%",
        root,
        capacity / 2**30,
        100 * args.high_watermark,
        100 * args.low_watermark,
    )
    try:
        while not stop.wait(args.interval_s):
            if not os.path.isdir(root):
                log.warning("%s was destroyed; restart the manager to publish a new generation", root)
                os.remove(ready_file)
                stop.wait()
                break
            evict_once(
                root,
                capacity,
                args.high_watermark,
                args.low_watermark,
                args.stale_tmp_s,
            )
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(ready_file)
    log.info("stopping; %s stays published", root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["serve", "create", "destroy", "evict", "status"])
    parser.add_argument("--base-dir", default=os.environ.get("KV_POOL_BASE_DIR", "/kv-pool"))
    parser.add_argument("--name", default=os.environ.get("KV_POOL_NAME", "default"))
    parser.add_argument(
        "--config",
        default=os.environ.get("KV_POOL_CONFIG", "{}"),
        help="JSON identity of the pool (e.g. model, block size, dtype); "
        "create refuses to reuse a generation with a different config",
    )
    parser.add_argument("--generation", help="generation to destroy")
    parser.add_argument(
        "--capacity-bytes",
        type=int,
        default=int(os.environ.get("KV_POOL_CAPACITY_BYTES", "0")),
        help="eviction capacity; 0 uses the size of the filesystem",
    )
    parser.add_argument(
        "--high-watermark", type=float, default=float(os.environ.get("KV_POOL_HIGH_WATERMARK", "0.9"))
    )
    parser.add_argument(
        "--low-watermark", type=float, default=float(os.environ.get("KV_POOL_LOW_WATERMARK", "0.75"))
    )
    parser.add_argument(
        "--interval-s", type=float, default=float(os.environ.get("KV_POOL_EVICT_INTERVAL_S", "2"))
    )
    parser.add_argument("--stale-tmp-s", type=float, default=60.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not 0 < args.low_watermark < args.high_watermark <= 1:
        parser.error("need 0 < --low-watermark < --high-watermark <= 1")
    config = json.loads(args.config)
    try:
        if args.command == "serve":
            serve(args, config)
        elif args.command == "create":
            print(json.dumps(create(args.base_dir, args.name, config)))
        elif args.command == "destroy":
            if not args.generation:
                parser.error("destroy requires --generation")
            destroy(args.base_dir, args.name, args.generation)
        elif args.command == "evict":
            root = os.path.join(args.base_dir, args.name)
            evict_once(
                root,
                capacity_of(root, args.capacity_bytes),
                args.high_watermark,
                args.low_watermark,
                args.stale_tmp_s,
            )
        elif args.command == "status":
            print(json.dumps(status(args.base_dir, args.name), indent=2))
    except PoolBusy as exc:
        log.error("%s", exc)
        return 3
    except PoolError as exc:
        log.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
