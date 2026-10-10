"""Fetch the public-dataset slices behind the `view`/`ask` corpus (plans/neuralese/VIEW_CORPUS.md).

Downloads are pinned to a revision, kept small (exact files or row slices, never whole corpora) and land in
one raw directory on the external drive with `sources.json`: per file its URL, revision, SHA-256, size and
licence. The builder (`view_corpus.py`) reads only that directory and local caches, never the network.

Sources fetched here:
- QMSum (Yale-LILY/QMSum, MIT): query-based meeting summarisation, `data/ALL/jsonl/{train,val,test}.jsonl`.
- FeTaQA (Yale-LILY/FeTaQA, CC-BY-SA-4.0): free-form table QA, `data/fetaQA-v1_{train,dev,test}.jsonl`.
- WikiTableQuestions (ppasupat/WikiTableQuestions, CC-BY-SA-4.0): repository tarball (TSV questions, CSV tables).
- CodeSearchNet Python (code-search-net/code_search_net, per-repository permissive licences): a row slice of
  the train parquet (whole leading row groups) and the full valid/test parquets.
- WebSRC v1.0 (X-LANCE/WebSRC_v1.0, CC-BY-4.0): per website `dataset.csv` and a bounded number of page HTML
  files, read member by member from the remote zip (screenshots and bounding boxes are not fetched).
- (v2) CodeSearchNet Go/Java/JavaScript/PHP/Ruby row slices, and the per-repository licence files the original
  release shipped (`<language>_licenses.pkl` inside the dataset's pre-Parquet zips, revision fdc6a9e): read member by
  member from the remote zip, unpickled by a restricted unpickler that refuses every class (plain containers and
  strings only), and stored as `codesearchnet/<language>-licenses.json` (repository -> licence files, detected SPDX,
  SHA-256 of each licence text). The pickles themselves are kept beside it for provenance.
- (v2) TabFact (wenhuchen/Table-Fact-Checking, data CC-BY-4.0 per the dataset card, code MIT): the collected
  statements (r1/r2), the official split ids, `table_to_page.json`, and the CSV tables of a seeded selection.
- (v3) BookSum (kmfoda/booksum chapter level: train/dev/test CSV), XSum (EdinburghNLP/xsum parquets), GovReport
  (ccdv/govreport-summarization parquets), five Mind2Web train JSON files (osunlp/Mind2Web; the test set is not
  fetched), LogHub (logpai/loghub: every system's 2k-line log with its structured CSV and templates), and a prefix of
  one Common Crawl WARC file (whole gzip members of the first FETCH_CC_BYTES bytes, by an HTTP range request).
  Multi-News, arXiv (S2ORC-parsed) and PubMed Central (Markdown) are read from local copies (view_corpus.py).
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training" / "neuralese"))
from natlang_neuralese.common.paths import resolve  # noqa: E402

DEFAULT_RAW = resolve("data_hdd", "natlang-development-data/data/neuralese/raw/view-sources-20261009")

GITHUB = {
    "qmsum": {"repo": "Yale-LILY/QMSum", "revision": "83d7768c1f2b4dfeb091385d3dc7e239b8e5bb7e", "license": "MIT",
              "files": ["data/ALL/jsonl/train.jsonl", "data/ALL/jsonl/val.jsonl", "data/ALL/jsonl/test.jsonl", "LICENSE"]},
    "fetaqa": {"repo": "Yale-LILY/FeTaQA", "revision": "bbc441be807212d736c8c35a18146b530dde11d0", "license": "CC-BY-SA-4.0",
               "files": ["data/fetaQA-v1_train.jsonl", "data/fetaQA-v1_dev.jsonl", "data/fetaQA-v1_test.jsonl", "LICENSE"]},
}
WTQ = {"repo": "ppasupat/WikiTableQuestions", "revision": "7d455a5a707b96341ef72aff9428749d443d8aa9", "license": "CC-BY-SA-4.0"}
CSN = {"repo": "code-search-net/code_search_net", "revision": "bd0cf261e357a3eb5c8fba490d23ec1a1cd59555",
       "license": "LicenseRef-repository-content",
       "files": {"train": "python/train-00000-of-00001.parquet", "validation": "python/validation-00000-of-00001.parquet",
                 "test": "python/test-00000-of-00001.parquet"}}
CSN_LANGUAGES = ("python", "go", "java", "javascript", "php", "ruby")
CSN_LEGACY = "fdc6a9e39575768c27eb8a2a5f702bf846eb4759"  # last revision with the original zips (licence pickles)
TABFACT = {"repo": "wenhuchen/Table-Fact-Checking", "revision": "2ab782ba42b5808076ac91fec846473aa5315a79",
           "license": "CC-BY-4.0",
           "files": ["collected_data/r1_training_all.json", "collected_data/r2_training_all.json", "data/train_id.json",
                     "data/val_id.json", "data/test_id.json", "data/table_to_page.json", "LICENSE"]}
V3_HF = {
    "booksum": {"repo": "kmfoda/booksum", "revision": "c62321036e5647db5767ecaff139912b554dc938", "license": "BSD-3-Clause",
                "files": ["train.csv", "dev.csv", "test.csv", "LICENSE.txt", "README.md"]},
    "xsum": {"repo": "EdinburghNLP/xsum", "revision": "7d4d486c2f8ef850b1a11aead99b894ff3dd7da9", "license": "LicenseRef-unknown",
             "files": ["data/train-00000-of-00001.parquet", "data/validation-00000-of-00001.parquet",
                       "data/test-00000-of-00001.parquet", "README.md"]},
    "govreport": {"repo": "ccdv/govreport-summarization", "revision": "4e21184e01ae8017e2c036e180fe5e541fef60a0",
                  "license": "LicenseRef-not-stated",
                  "files": ["document/train-00000-of-00002.parquet", "document/train-00001-of-00002.parquet",
                            "document/validation-00000-of-00001.parquet", "document/test-00000-of-00001.parquet", "README.md"]},
    "mind2web": {"repo": "osunlp/Mind2Web", "revision": "17ece8eb89862368edc0cc806acee6fca5163474", "license": "CC-BY-4.0",
                 "files": ["data/train/train_10.json", "data/train/train_6.json", "data/train/train_0.json", "data/train/train_2.json",
                           "data/train/train_8.json", "README.md"]},
}
LOGHUB = {"repo": "logpai/loghub", "revision": "dd61d0952749ee7963bde24220d1be5ede023033", "license": "LicenseRef-loghub",
          "systems": ["Android", "Apache", "BGL", "HDFS", "HPC", "Hadoop", "HealthApp", "Linux", "Mac", "OpenSSH",
                      "OpenStack", "Proxifier", "Spark", "Thunderbird", "Windows", "Zookeeper"]}
COMMON_CRAWL = {"crawl": "CC-MAIN-2025-38", "license": "LicenseRef-commoncrawl-terms",
                "path": "crawl-data/CC-MAIN-2025-38/segments/1757047532641.17/warc/"
                        "CC-MAIN-20250905112101-20250905142101-00000.warc.gz"}
FETCH_CC_BYTES = 256 << 20
WEBSRC = {"repo": "X-LANCE/WebSRC_v1.0", "revision": "7aa0bc6efc7ef43f68c192e2091108541acbaf1a", "license": "CC-BY-4.0",
          "file": "WebSRC_v1.0_train+dev.zip"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _get(url: str, attempts: int = 4) -> bytes:
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "natlang-view-corpus"}), timeout=120) as r:
                return r.read()
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".pending")
    tmp.write_bytes(data)
    tmp.replace(path)


def _entry(raw: Path, path: Path, **extra) -> dict:
    return {"path": str(path.relative_to(raw)), "bytes": path.stat().st_size, "sha256": _sha256(path), **extra}


def fetch_github(raw: Path, log=print) -> list[dict]:
    out = []
    for name, spec in GITHUB.items():
        for f in spec["files"]:
            url = f"https://raw.githubusercontent.com/{spec['repo']}/{spec['revision']}/{f}"
            dest = raw / name / f
            if not dest.exists():
                _write(dest, _get(url))
            out.append(_entry(raw, dest, dataset=name, url=url, upstream=f"github:{spec['repo']}",
                              revision=spec["revision"], license=spec["license"]))
            log(f"{name}: {f} {dest.stat().st_size} bytes")
    return out


def fetch_wtq(raw: Path, log=print) -> list[dict]:
    """The WikiTableQuestions tarball, unpacked to TSV question files, CSV tables and the licence."""
    url = f"https://codeload.github.com/{WTQ['repo']}/tar.gz/{WTQ['revision']}"
    root = raw / "wtq"
    tarball = root / "repo.tar.gz"
    if not tarball.exists():
        _write(tarball, _get(url))
    out = [_entry(raw, tarball, dataset="wtq", url=url, upstream=f"github:{WTQ['repo']}", revision=WTQ["revision"],
                  license=WTQ["license"])]
    unpacked = root / "repo"
    if not unpacked.exists():
        with tarfile.open(tarball) as tar:
            members = []
            for m in tar.getmembers():
                rel = m.name.split("/", 1)[-1]
                if not m.isfile() or not (rel.startswith("data/") or rel.startswith("csv/") or rel in ("LICENSE", "README.md")):
                    continue
                if rel.startswith("data/") and not rel.endswith(".tsv"):
                    continue
                if rel.startswith("csv/") and not rel.endswith(".csv"):
                    continue
                m.name = rel
                members.append(m)
            tmp = root / "repo.pending"
            tar.extractall(tmp, members=members, filter="data")
            tmp.replace(unpacked)
    log(f"wtq: {sum(1 for _ in unpacked.rglob('*.csv'))} csv tables")
    return out


def _hf_fs():
    from huggingface_hub import HfFileSystem
    return HfFileSystem()


def fetch_csn(raw: Path, train_rows: int, log=print, languages=("python",), lang_train_rows: int | None = None) -> list[dict]:
    """CodeSearchNet per language: whole leading row groups of train up to `train_rows` (other languages:
    `lang_train_rows`), plus valid and test. Python keeps its v1 file names (`python-<split>.parquet`)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    fs = _hf_fs()
    out = []
    for lang in languages:
        limit = train_rows if lang == "python" else (lang_train_rows or train_rows)
        for split in ("train", "validation", "test"):
            rel = f"{lang}/{split}-00000-of-00001.parquet"
            dest = raw / "codesearchnet" / f"{lang}-{split}.parquet"
            hf_path = f"datasets/{CSN['repo']}@{CSN['revision']}/{rel}"
            if dest.exists() and split == "train" and pq.ParquetFile(dest).metadata.num_rows < limit:
                dest.unlink()  # a larger slice is wanted: refetch (the leading row groups are a prefix)
            if not dest.exists():
                with fs.open(hf_path, "rb", block_size=8 << 20) as remote:
                    pf = pq.ParquetFile(remote)
                    if split == "train":
                        tables, rows = [], 0
                        for g in range(pf.metadata.num_row_groups):
                            tables.append(pf.read_row_group(g))
                            rows += tables[-1].num_rows
                            if rows >= limit:
                                break
                        table = pa.concat_tables(tables)
                    else:
                        table = pf.read()
                dest.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(table, str(dest) + ".pending")
                Path(str(dest) + ".pending").replace(dest)
            note = "whole file"
            if split == "train":
                note = f"leading row groups of the train file, {pq.ParquetFile(dest).metadata.num_rows} rows"
            out.append(_entry(raw, dest, dataset="codesearchnet", language=lang, url=f"https://huggingface.co/{hf_path}",
                              upstream=f"hf:{CSN['repo']}", revision=CSN["revision"], license=CSN["license"], notes=note))
            log(f"codesearchnet {lang} {split}: {pq.ParquetFile(dest).metadata.num_rows} rows")
    return out


class _NoClasses:
    """A restricted unpickler: plain containers, strings and numbers only; any class reference is refused."""

    @staticmethod
    def load(data: bytes):
        import pickle

        class Safe(pickle.Unpickler):
            def find_class(self, module, name):
                raise pickle.UnpicklingError(f"refused class {module}.{name}")

        return Safe(io.BytesIO(data)).load()


def fetch_csn_licenses(raw: Path, languages=CSN_LANGUAGES, log=print) -> list[dict]:
    """The original release's per-repository licence files, as JSON with a detected SPDX per repository."""
    from .view_extract import detect_license

    fs = _hf_fs()
    out = []
    for lang in languages:
        pkl = raw / "codesearchnet" / f"{lang}_licenses.pkl"
        hf_path = f"datasets/{CSN['repo']}@{CSN_LEGACY}/data/{lang}.zip"
        if not pkl.exists():
            with fs.open(hf_path, "rb", block_size=4 << 20) as remote, zipfile.ZipFile(remote) as z:
                _write(pkl, z.read(f"{lang}_licenses.pkl"))
        dest = raw / "codesearchnet" / f"{lang}-licenses.json"
        if not dest.exists():
            table = _NoClasses.load(pkl.read_bytes())
            repos = {}
            for repo, files in sorted(table.items()):
                entries = [(str(p), str(t)) for p, t in files]
                spdx = next((d for d in (detect_license(t) for _, t in entries) if d), None)
                repos[str(repo)] = {"spdx": spdx, "files": [p for p, _ in entries],
                                    "sha256": [hashlib.sha256(t.encode("utf-8", "surrogatepass")).hexdigest() for _, t in entries]}
            _write(dest, (json.dumps(repos, indent=0, sort_keys=True) + "\n").encode("utf-8"))
        url = f"https://huggingface.co/{hf_path}#{lang}_licenses.pkl"
        out.append(_entry(raw, pkl, dataset="codesearchnet-licenses", language=lang, url=url, upstream=f"hf:{CSN['repo']}",
                          revision=CSN_LEGACY, license=CSN["license"], notes="original release licence files per repository"))
        out.append(_entry(raw, dest, dataset="codesearchnet-licenses", language=lang, url=url, upstream=f"hf:{CSN['repo']}",
                          revision=CSN_LEGACY, license=CSN["license"],
                          notes="derived: restricted unpickle, licence detected by view_extract.detect_license"))
        log(f"codesearchnet licences {lang}: {dest.stat().st_size} bytes")
    return out


def fetch_tabfact(raw: Path, tables_per_split: dict, log=print) -> list[dict]:
    """TabFact statements, split ids, page titles and a seeded selection of tables (hash order per split)."""
    out = []
    base = f"https://raw.githubusercontent.com/{TABFACT['repo']}/{TABFACT['revision']}"
    for f in TABFACT["files"]:
        dest = raw / "tabfact" / f
        if not dest.exists():
            _write(dest, _get(f"{base}/{f}"))
        out.append(_entry(raw, dest, dataset="tabfact", url=f"{base}/{f}", upstream=f"github:{TABFACT['repo']}",
                          revision=TABFACT["revision"], license=TABFACT["license"]))
    statements = {}
    for f in ("collected_data/r1_training_all.json", "collected_data/r2_training_all.json"):
        statements.update(json.loads((raw / "tabfact" / f).read_text()))
    for split, name in (("train", "train_id.json"), ("validation", "val_id.json"), ("test", "test_id.json")):
        ids = [t for t in json.loads((raw / "tabfact/data" / name).read_text()) if t in statements]
        ids.sort(key=lambda t: hashlib.sha256(f"tabfact\x1f{t}".encode()).hexdigest())
        got = 0
        for t in ids[: tables_per_split.get(split, 0)]:
            dest = raw / "tabfact/data/all_csv" / t
            url = f"{base}/data/all_csv/{t}"
            if not dest.exists():
                try:
                    _write(dest, _get(url))
                except OSError as error:
                    log(f"tabfact {t}: {error}")
                    continue
            out.append(_entry(raw, dest, dataset="tabfact", url=url, upstream=f"github:{TABFACT['repo']}",
                              revision=TABFACT["revision"], license=TABFACT["license"]))
            got += 1
        log(f"tabfact {split}: {got} tables")
    return out


def fetch_websrc(raw: Path, pages_per_site: int, log=print) -> list[dict]:
    """Per website: dataset.csv and the first `pages_per_site` HTML pages (by page id) that have questions."""
    fs = _hf_fs()
    hf_path = f"datasets/{WEBSRC['repo']}@{WEBSRC['revision']}/{WEBSRC['file']}"
    root = raw / "websrc"
    out = []
    with fs.open(hf_path, "rb", block_size=4 << 20) as remote, zipfile.ZipFile(remote) as z:
        names = z.namelist()
        csvs = sorted(n for n in names if n.endswith("/dataset.csv"))
        html = {n for n in names if n.endswith(".html") and "/processed_data/" in n}
        for csv_name in csvs:
            site_dir = csv_name[: -len("dataset.csv")]
            dest_csv = root / csv_name
            if not dest_csv.exists():
                _write(dest_csv, z.read(csv_name))
            out.append(_entry(raw, dest_csv, dataset="websrc", url=f"https://huggingface.co/{hf_path}#{csv_name}",
                              upstream=f"hf:{WEBSRC['repo']}", revision=WEBSRC["revision"], license=WEBSRC["license"]))
            import csv as _csv
            with open(dest_csv, encoding="utf-8", newline="") as stream:
                page_ids = sorted({row["id"][2:9] for row in _csv.DictReader(stream) if row.get("id")})
            for page in page_ids[:pages_per_site]:
                member = f"{site_dir}processed_data/{page}.html"
                if member not in html:
                    continue
                dest = root / member
                if not dest.exists():
                    _write(dest, z.read(member))
                out.append(_entry(raw, dest, dataset="websrc", url=f"https://huggingface.co/{hf_path}#{member}",
                                  upstream=f"hf:{WEBSRC['repo']}", revision=WEBSRC["revision"], license=WEBSRC["license"]))
            log(f"websrc {site_dir}: {min(len(page_ids), pages_per_site)} pages")
    return out


def _download(url: str, dest: Path, *, limit: int | None = None, attempts: int = 4) -> None:
    """Stream `url` to `dest` (atomically); `limit` fetches only the first `limit` bytes (HTTP range)."""
    headers = {"User-Agent": "natlang-view-corpus"}
    if limit:
        headers["Range"] = f"bytes=0-{limit - 1}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".pending")
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=300) as r, open(tmp, "wb") as out:
                for block in iter(lambda: r.read(8 << 20), b""):
                    out.write(block)
            tmp.replace(dest)
            return
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)


def fetch_v3(raw: Path, log=print) -> list[dict]:
    """The v3 sources: BookSum, XSum, GovReport and Mind2Web files from the Hugging Face Hub, LogHub from GitHub,
    and a Common Crawl WARC prefix."""
    out = []
    for name, spec in V3_HF.items():
        for f in spec["files"]:
            url = f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}/{f}"
            dest = raw / name / f
            if not dest.exists():
                _download(url, dest)
            out.append(_entry(raw, dest, dataset=name, url=url, upstream=f"hf:{spec['repo']}", revision=spec["revision"],
                              license=spec["license"]))
            log(f"{name}: {f} {dest.stat().st_size} bytes")
    base = f"https://raw.githubusercontent.com/{LOGHUB['repo']}/{LOGHUB['revision']}"
    for f in ["LICENSE", "README.md"] + [f"{s}/{s}_2k.{ext}" for s in LOGHUB["systems"]
                                         for ext in ("log", "log_structured.csv", "log_templates.csv")]:
        dest = raw / "loghub" / f
        if not dest.exists():
            _write(dest, _get(f"{base}/{f}"))
        out.append(_entry(raw, dest, dataset="loghub", url=f"{base}/{f}", upstream=f"github:{LOGHUB['repo']}",
                          revision=LOGHUB["revision"], license=LOGHUB["license"]))
    log(f"loghub: {len(LOGHUB['systems'])} systems")
    url = f"https://data.commoncrawl.org/{COMMON_CRAWL['path']}"
    dest = raw / "common_crawl" / Path(COMMON_CRAWL["path"]).name.replace(".warc.gz", f".first{FETCH_CC_BYTES}.warc.gz")
    if not dest.exists():
        _download(url, dest, limit=FETCH_CC_BYTES)
    out.append(_entry(raw, dest, dataset="common_crawl", url=url, upstream=f"commoncrawl:{COMMON_CRAWL['crawl']}",
                      revision=COMMON_CRAWL["crawl"], license=COMMON_CRAWL["license"],
                      notes=f"bytes 0-{FETCH_CC_BYTES - 1} of the WARC file (whole gzip members are read; the cut one is skipped)"))
    log(f"common_crawl: {dest.stat().st_size} bytes")
    return out


def fetch(raw: Path = DEFAULT_RAW, *, csn_train_rows: int = 20000, websrc_pages_per_site: int = 12,
          csn_languages=("python",), csn_lang_train_rows: int | None = None, csn_licenses: bool = False,
          tabfact_tables: dict | None = None, v3: bool = False, log=print) -> dict:
    """v1 used the defaults; v2 adds the other CodeSearchNet languages, their licence files and TabFact; v3 adds
    BookSum, XSum, GovReport, Mind2Web, LogHub and a Common Crawl WARC prefix."""
    raw.mkdir(parents=True, exist_ok=True)
    files = []
    files += fetch_github(raw, log)
    files += fetch_wtq(raw, log)
    files += fetch_csn(raw, csn_train_rows, log, languages=csn_languages, lang_train_rows=csn_lang_train_rows)
    if csn_licenses:
        files += fetch_csn_licenses(raw, csn_languages, log)
    files += fetch_websrc(raw, websrc_pages_per_site, log)
    if tabfact_tables:
        files += fetch_tabfact(raw, tabfact_tables, log)
    if v3:
        files += fetch_v3(raw, log)
    report = {"schema": "natlang.view-corpus-sources/1", "raw": str(raw), "files": files,
              "options": {"csn_train_rows": csn_train_rows, "websrc_pages_per_site": websrc_pages_per_site,
                          "csn_languages": list(csn_languages), "csn_lang_train_rows": csn_lang_train_rows,
                          "csn_licenses": csn_licenses, "tabfact_tables": tabfact_tables, "v3": v3}}
    _write(raw / "sources.json", (json.dumps(report, indent=1, sort_keys=True) + "\n").encode())  # never in place: raw
    # directories start as hard-linked copies of their predecessor
    return report
