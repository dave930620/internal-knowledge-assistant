# =============================================================================
# scripts/download_raw.py
#
# PURPOSE:
#   Fetches raw HTML pages from external engineering documentation sources
#   and saves them to disk. Also writes a manifest (JSONL) record for each
#   successfully downloaded file so downstream steps know what was collected.
#
# INPUT:
#   Source URL lists (plain .txt files, one URL per line):
#     data/raw/manifests/source_lists/aws_engineering_urls.txt
#     data/raw/manifests/source_lists/cloudflare_engineering_urls.txt
#     data/raw/manifests/source_lists/kubernetes_docs_urls.txt
#     data/raw/manifests/source_lists/stripe_docs_urls.txt
#     data/raw/manifests/source_lists/ops_urls.txt
#
# OUTPUT:
#   Raw HTML files saved by source group:
#     data/raw/engineering_blogs/aws/eng_aws_NNN.html
#     data/raw/engineering_blogs/cloudflare/eng_cf_NNN.html
#     data/raw/official_docs/kubernetes/doc_k8s_NNN.html
#     data/raw/official_docs/stripe/doc_stripe_NNN.html
#     data/raw/ops_troubleshooting/mixed_ops/ops_mix_NNN.html
#
#   Manifest JSONL files (one record per downloaded file):
#     data/raw/manifests/engineering_blogs_manifest.jsonl
#     data/raw/manifests/official_docs_manifest.jsonl
#     data/raw/manifests/ops_manifest.jsonl
#
#   Failure log (records URLs that could not be fetched):
#     data/raw/manifests/failed_urls.jsonl
# =============================================================================

import json
import re
from pathlib import Path
from urllib.parse import urlparse

import requests

BASE_DIR = Path(__file__).resolve().parent.parent

SOURCE_LISTS = [
    {
        "name": "aws_engineering",
        "txt_path": BASE_DIR / "data/raw/manifests/source_lists/aws_engineering_urls.txt",
        "save_dir": BASE_DIR / "data/raw/engineering_blogs/aws",
        "manifest_path": BASE_DIR / "data/raw/manifests/engineering_blogs_manifest.jsonl",
        "doc_prefix": "eng_aws",
        "source_type": "engineering_blog",
        "provider": "aws",
        "allowed_domains": {"aws.amazon.com"},
        "allowed_url_keywords": ["/blogs/architecture/"],
    },
    {
        "name": "cloudflare_engineering",
        "txt_path": BASE_DIR / "data/raw/manifests/source_lists/cloudflare_engineering_urls.txt",
        "save_dir": BASE_DIR / "data/raw/engineering_blogs/cloudflare",
        "manifest_path": BASE_DIR / "data/raw/manifests/engineering_blogs_manifest.jsonl",
        "doc_prefix": "eng_cf",
        "source_type": "engineering_blog",
        "provider": "cloudflare",
        "allowed_domains": {"blog.cloudflare.com"},
        "allowed_url_keywords": [],
    },
    {
        "name": "kubernetes_docs",
        "txt_path": BASE_DIR / "data/raw/manifests/source_lists/kubernetes_docs_urls.txt",
        "save_dir": BASE_DIR / "data/raw/official_docs/kubernetes",
        "manifest_path": BASE_DIR / "data/raw/manifests/official_docs_manifest.jsonl",
        "doc_prefix": "doc_k8s",
        "source_type": "official_doc",
        "provider": "kubernetes",
        "allowed_domains": {"kubernetes.io"},
        "allowed_url_keywords": ["/docs/"],
    },
    {
        "name": "stripe_docs",
        "txt_path": BASE_DIR / "data/raw/manifests/source_lists/stripe_docs_urls.txt",
        "save_dir": BASE_DIR / "data/raw/official_docs/stripe",
        "manifest_path": BASE_DIR / "data/raw/manifests/official_docs_manifest.jsonl",
        "doc_prefix": "doc_stripe",
        "source_type": "official_doc",
        "provider": "stripe",
        "allowed_domains": {"docs.stripe.com"},
        "allowed_url_keywords": [],
    },
    {
        "name": "ops_mixed",
        "txt_path": BASE_DIR / "data/raw/manifests/source_lists/ops_urls.txt",
        "save_dir": BASE_DIR / "data/raw/ops_troubleshooting/mixed_ops",
        "manifest_path": BASE_DIR / "data/raw/manifests/ops_manifest.jsonl",
        "doc_prefix": "ops_mix",
        "source_type": "ops_troubleshooting",
        "provider": "mixed",
        "allowed_domains": {"aws.amazon.com", "kubernetes.io"},
        "allowed_url_keywords": [],
    },
]

FAILED_LOG_PATH = BASE_DIR / "data/raw/manifests/failed_urls.jsonl"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; corpus-builder/1.0)"
}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def read_urls(txt_path: Path):
    if not txt_path.exists():
        print(f"[WARN] URL list not found: {txt_path}")
        return []

    with open(txt_path, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    return urls


def append_jsonl(path: Path, record: dict):
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_existing_urls(manifest_path: Path):
    existing_urls = set()
    if not manifest_path.exists():
        return existing_urls

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                url = obj.get("source_url")
                if url:
                    existing_urls.add(url)
            except json.JSONDecodeError:
                continue

    return existing_urls


def get_next_index(save_dir: Path, prefix: str):
    ensure_dir(save_dir)
    existing = list(save_dir.glob(f"{prefix}_*.html"))
    max_num = 0

    for p in existing:
        stem = p.stem
        parts = stem.split("_")
        try:
            num = int(parts[-1])
            max_num = max(max_num, num)
        except ValueError:
            continue

    return max_num + 1


def normalize_url(url: str):
    return url.strip()


def is_url_allowed(url: str, config: dict):
    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    if domain.startswith("www."):
        domain = domain[4:]

    allowed_domains = config.get("allowed_domains", set())
    if allowed_domains and domain not in allowed_domains:
        return False, f"domain '{domain}' not allowed for source '{config['name']}'"

    allowed_keywords = config.get("allowed_url_keywords", [])
    if allowed_keywords:
        if not any(keyword in url for keyword in allowed_keywords):
            return False, f"url path not matched for source '{config['name']}'"

    return True, ""


def fetch_html(url: str):
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return resp.text


def extract_title(html: str):
    match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    title = match.group(1)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def log_failure(url: str, config: dict, reason: str):
    record = {
        "source_name": config["name"],
        "source_url": url,
        "reason": reason,
    }
    append_jsonl(FAILED_LOG_PATH, record)


def process_source(config: dict):
    print("\n" + "=" * 80)
    print(f"[SOURCE] {config['name']}")
    print(f"[TXT]    {config['txt_path']}")
    print(f"[SAVE]   {config['save_dir']}")
    print(f"[MANIF]  {config['manifest_path']}")
    print("=" * 80)

    urls = read_urls(config["txt_path"])
    if not urls:
        print(f"[INFO] No URLs in {config['txt_path']}")
        return

    ensure_dir(config["save_dir"])
    existing_urls = load_existing_urls(config["manifest_path"])
    next_idx = get_next_index(config["save_dir"], config["doc_prefix"])

    for raw_url in urls:
        url = normalize_url(raw_url)

        allowed, reason = is_url_allowed(url, config)
        if not allowed:
            print(f"[SKIP] {url} | {reason}")
            log_failure(url, config, f"URL validation failed: {reason}")
            continue

        if url in existing_urls:
            print(f"[SKIP] Already downloaded: {url}")
            continue

        doc_id = f"{config['doc_prefix']}_{next_idx:03d}"
        save_path = config["save_dir"] / f"{doc_id}.html"

        try:
            html = fetch_html(url)

            with open(save_path, "w", encoding="utf-8") as f:
                f.write(html)

            title = extract_title(html)

            record = {
                "doc_id": doc_id,
                "title": title,
                "source_type": config["source_type"],
                "provider": config["provider"],
                "source_url": url,
                "raw_path": str(save_path.relative_to(BASE_DIR)),
                "content_format": "html",
            }

            append_jsonl(config["manifest_path"], record)
            existing_urls.add(url)

            print(f"[OK]   {doc_id} <- {url}")
            next_idx += 1

        except requests.HTTPError as e:
            reason = f"HTTP error: {e}"
            print(f"[FAIL] {url} | {reason}")
            log_failure(url, config, reason)

        except requests.RequestException as e:
            reason = f"Request error: {e}"
            print(f"[FAIL] {url} | {reason}")
            log_failure(url, config, reason)

        except Exception as e:
            reason = f"Unexpected error: {e}"
            print(f"[FAIL] {url} | {reason}")
            log_failure(url, config, reason)


def main():
    for config in SOURCE_LISTS:
        process_source(config)


if __name__ == "__main__":
    main()