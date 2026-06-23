import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlencode, urlparse
from urllib.request import Request, urlopen


COMMONS_API = "https://commons.wikimedia.org/w/api.php"
DEFAULT_KEYWORDS = ["japanese sign", "menu", "street sign", "book page"]
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)


def fetch_json(params, max_retries=3, retry_backoff=2.0):
    url = f"{COMMONS_API}?{urlencode(params)}"
    last_error = None

    for attempt in range(max_retries + 1):
        if attempt > 0:
            time.sleep(retry_backoff ** attempt)

        req = Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(req, timeout=20) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code in {429, 502, 503, 504}:
                last_error = error
                retry_after = error.headers.get("Retry-After")
                if retry_after:
                    try:
                        time.sleep(int(retry_after))
                    except ValueError:
                        pass
                continue
            raise
        except URLError as error:
            last_error = error
            continue

    raise last_error if last_error else RuntimeError("Failed to fetch data")


def search_commons(keyword, limit, offset, max_retries, retry_backoff):
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": keyword,
        "gsrnamespace": 6,
        "gsrlimit": min(limit, 50),
        "gsroffset": offset,
        "prop": "imageinfo",
        "iiprop": "url|size|mime",
        "format": "json",
    }
    data = fetch_json(params, max_retries=max_retries, retry_backoff=retry_backoff)
    pages = data.get("query", {}).get("pages", {})
    results = []

    for page in pages.values():
        title = page.get("title")
        imageinfo = page.get("imageinfo", [])
        if not imageinfo:
            continue
        info = imageinfo[0]
        results.append(
            {
                "title": title,
                "url": info.get("url"),
                "mime": info.get("mime"),
                "width": info.get("width"),
                "height": info.get("height"),
                "pageid": page.get("pageid"),
            }
        )

    next_offset = data.get("continue", {}).get("gsroffset")
    return results, next_offset


def normalize_filename(text):
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", text)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "image"


def choose_filename(url, title):
    parsed = urlparse(url)
    name = unquote(Path(parsed.path).name)
    if not name:
        name = title.replace("File:", "") if title else "image"
    return normalize_filename(name)


def download_with_hash(url, dest_path):
    req = Request(url, headers={"User-Agent": USER_AGENT})
    sha1 = hashlib.sha1()
    with urlopen(req, timeout=30) as response, open(dest_path, "wb") as output:
        while True:
            chunk = response.read(1024 * 64)
            if not chunk:
                break
            output.write(chunk)
            sha1.update(chunk)
    return sha1.hexdigest()


def is_valid_image(item, min_width, min_height, allowed_mime):
    if not item.get("url"):
        return False
    if allowed_mime and item.get("mime") not in allowed_mime:
        return False
    if (item.get("width") or 0) < min_width:
        return False
    if (item.get("height") or 0) < min_height:
        return False
    return True


def ensure_unique_path(path):
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(1, 9999):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError("Failed to find unique filename")


def main():
    parser = argparse.ArgumentParser(
        description="Download OCR-friendly images from Wikimedia Commons"
    )
    parser.add_argument("keywords", nargs="*", default=DEFAULT_KEYWORDS)
    parser.add_argument("--limit-per-keyword", type=int, default=30)
    parser.add_argument("--min-width", type=int, default=640)
    parser.add_argument("--min-height", type=int, default=640)
    parser.add_argument(
        "--allowed-mime",
        default="image/jpeg,image/png,image/webp",
        help="Comma-separated list of allowed MIME types",
    )
    parser.add_argument("--request-delay", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff", type=float, default=2.0)

    args = parser.parse_args()
    allowed_mime = [
        item.strip() for item in args.allowed_mime.split(",") if item.strip()
    ]

    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "output" / "commons_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_file = base_dir / "output" / "commons_images.json"

    seen_urls = set()
    seen_hashes = set()
    collected = []

    for keyword in args.keywords:
        fetched = 0
        offset = 0
        while fetched < args.limit_per_keyword:
            batch_size = min(50, args.limit_per_keyword - fetched)
            items, next_offset = search_commons(
                keyword,
                batch_size,
                offset,
                args.max_retries,
                args.retry_backoff,
            )
            if not items:
                break

            for item in items:
                if fetched >= args.limit_per_keyword:
                    break

                if not is_valid_image(
                    item, args.min_width, args.min_height, allowed_mime
                ):
                    continue

                url = item["url"]
                if url in seen_urls:
                    continue

                filename = choose_filename(url, item.get("title") or "image")
                file_path = ensure_unique_path(output_dir / filename)

                try:
                    sha1 = download_with_hash(url, file_path)
                except Exception:
                    if file_path.exists():
                        file_path.unlink()
                    continue

                if sha1 in seen_hashes:
                    file_path.unlink(missing_ok=True)
                    continue

                seen_urls.add(url)
                seen_hashes.add(sha1)
                collected.append(
                    {
                        "source": "wikimedia_commons",
                        "keyword": keyword,
                        "title": item.get("title"),
                        "url": url,
                        "mime": item.get("mime"),
                        "width": item.get("width"),
                        "height": item.get("height"),
                        "path": str(file_path),
                        "sha1": sha1,
                    }
                )
                fetched += 1

            if next_offset is None:
                break
            offset = next_offset
            time.sleep(args.request_delay)

    with open(meta_file, "w", encoding="utf-8") as output:
        json.dump(
            {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "count": len(collected),
                "items": collected,
            },
            output,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved {len(collected)} images to {output_dir}")
    print(f"Metadata written to {meta_file}")


if __name__ == "__main__":
    main()
