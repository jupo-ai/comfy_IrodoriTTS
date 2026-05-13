from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from huggingface_hub import hf_hub_url, list_repo_files


def _format_path(path: str | Path | None) -> str:
    if path is None:
        return "(default cache)"
    return str(Path(path))


def hf_progress_bars_enabled() -> bool:
    value = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS", "")
    return value.strip().lower() not in {"1", "on", "true", "yes"}


def progress_label(kind: str) -> str:
    if kind == "tqdm":
        return "tqdm"
    if kind == "api":
        return "api"
    if kind == "local":
        return "local"
    if kind == "huggingface":
        return "huggingface/tqdm" if hf_progress_bars_enabled() else "huggingface/disabled"
    return kind


def repo_cache_dir(cache_dir: str | Path, repo_id: str) -> Path:
    safe_name = str(repo_id).replace("/", "_").replace("\\", "_")
    path = Path(cache_dir) / safe_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_hf_file_with_progress(
    repo_id: str,
    filename: str,
    target_dir: str | Path,
    *,
    item_type: str,
    timeout: int = 30,
) -> Path:
    target_dir = Path(target_dir)
    target = target_dir / filename
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)

    import requests
    from tqdm.auto import tqdm

    url = hf_hub_url(repo_id=repo_id, filename=filename)
    tmp_path = target.with_name(target.name + ".tmp")
    with download_log(item_type, f"{repo_id}/{filename}", cache_dir=target_dir, progress="tqdm"):
        try:
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length") or 0) or None
                desc = f"[download] {item_type} {repo_id}/{filename}"
                with open(tmp_path, "wb") as f, tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=desc,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        pbar.update(len(chunk))
            tmp_path.replace(target)
        except Exception:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise
    return target


_TOKENIZER_FILENAMES = {
    "added_tokens.json",
    "config.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
}


def _looks_like_tokenizer_cache(path: Path) -> bool:
    has_tokenizer_model = any(
        (path / name).exists()
        for name in (
            "sentencepiece.bpe.model",
            "spiece.model",
            "tokenizer.json",
            "tokenizer.model",
            "vocab.json",
            "vocab.txt",
        )
    )
    return has_tokenizer_model and (path / "tokenizer_config.json").exists()


def download_hf_tokenizer_with_progress(repo_id: str, cache_dir: str | Path) -> Path:
    target_dir = repo_cache_dir(cache_dir, repo_id)
    if _looks_like_tokenizer_cache(target_dir):
        return target_dir

    with download_log("tokenizer", f"{repo_id} file list", cache_dir=target_dir, progress="api"):
        filenames = [
            name
            for name in list_repo_files(repo_id)
            if "/" not in name and name in _TOKENIZER_FILENAMES
        ]
    if not filenames:
        raise RuntimeError(f"No tokenizer files found in Hugging Face repo: {repo_id}")

    for filename in filenames:
        download_hf_file_with_progress(
            repo_id,
            filename,
            target_dir,
            item_type="tokenizer",
        )
    return target_dir


@contextmanager
def download_log(
    item_type: str,
    source: str,
    *,
    cache_dir: str | Path | None = None,
    progress: str = "huggingface",
) -> Iterator[None]:
    started_at = time.perf_counter()
    print(
        (
            f"[download] {item_type}: {source} -> {_format_path(cache_dir)} "
            f"(progress={progress_label(progress)})"
        ),
        flush=True,
    )
    try:
        yield
    except Exception as exc:
        elapsed = time.perf_counter() - started_at
        print(
            f"[download] {item_type}: failed after {elapsed:.1f}s: {exc}",
            flush=True,
        )
        raise
    else:
        elapsed = time.perf_counter() - started_at
        print(f"[download] {item_type}: done in {elapsed:.1f}s", flush=True)
