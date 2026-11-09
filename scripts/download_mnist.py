"""Download MNIST into the torchvision raw layout from the ossci mirror.

Usage: python scripts/download_mnist.py [dest_root]
Default dest_root: <repo>/data/mnist  (matches utils/env.MNIST_PATH)
Files land in <dest_root>/MNIST/raw/*.gz, which is what torchvision.datasets.MNIST
expects, so datasets can then be constructed with download=False.
"""

from __future__ import annotations

import gzip
import hashlib
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
# MD5s of the mirrored (ossci) files, per torchvision's mirrored_md5s list.
FILES = {
    "train-images-idx3-ubyte.gz": ("f68b3c2dcbeaaa9fbdd348bbdeb94873", 9912422),
    "train-labels-idx1-ubyte.gz": ("d53e105ee54ea40749a09fcbcd1e9432", 28881),
    "t10k-images-idx3-ubyte.gz": ("9fb629c4189551a2d022fa330f9573f3", 1648877),
    "t10k-labels-idx1-ubyte.gz": ("ec29112dd5afa0611ce80d1b7f02629c", 4542),
}


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else repo_root / "data" / "mnist"
    raw = dest / "MNIST" / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    ok = True
    for name, (expect_md5, expect_size) in FILES.items():
        target = raw / name
        if target.is_file() and md5sum(target) == expect_md5:
            print(f"[skip] {name}: already present and verified")
            continue
        url = BASE_URL + name
        print(f"[get ] {name} <- {url}")
        tmp = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        got_md5 = md5sum(tmp)
        got_size = tmp.stat().st_size
        if got_md5 != expect_md5:
            print(f"[warn] {name}: md5 mismatch (got {got_md5}, want {expect_md5}); size {got_size}/{expect_size}")
        else:
            print(f"[ok  ] {name}: verified ({got_size} bytes)")
        tmp.rename(target)
        ok = ok and got_md5 == expect_md5

    # Sanity-check the idx structure (magic + counts) straight from the gz files.
    import struct

    with gzip.open(raw / "train-images-idx3-ubyte.gz", "rb") as f:
        magic, n, r, c = struct.unpack(">4I", f.read(16))
        assert magic == 2051 and (n, r, c) == (60000, 28, 28), (magic, n, r, c)
        print(f"[chk ] train images: {n} x {r}x{c}")
    with gzip.open(raw / "train-labels-idx1-ubyte.gz", "rb") as f:
        magic, n = struct.unpack(">2I", f.read(8))
        assert magic == 2049 and n == 60000, (magic, n)
        print(f"[chk ] train labels: {n}")
    with gzip.open(raw / "t10k-images-idx3-ubyte.gz", "rb") as f:
        magic, n, r, c = struct.unpack(">4I", f.read(16))
        assert magic == 2051 and (n, r, c) == (10000, 28, 28), (magic, n, r, c)
        print(f"[chk ] test images: {n} x {r}x{c}")

    try:
        from torchvision.datasets import MNIST as TVMNIST

        ds = TVMNIST(root=str(dest), train=True, download=False)
        img, label = ds[0]
        print(f"[tv  ] torchvision MNIST loaded, len={len(ds)}, sample={img.size}, label={label}")
    except ImportError:
        print("[tv  ] torchvision not installed yet; idx-level checks passed, rerun later to confirm")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
