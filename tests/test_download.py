"""Tests del script de descarga (sin tocar la red)."""

import hashlib

from src.data.download import build_manifest_entries, sha256_of


def test_sha256_of_matches_hashlib(tmp_path):
    f = tmp_path / "sample.txt"
    content = b"human or machine?" * 1000
    f.write_bytes(content)
    assert sha256_of(f) == hashlib.sha256(content).hexdigest()


def test_build_manifest_entries_lists_all_files(tmp_path):
    dest = tmp_path / "raw" / "daigt_test"
    (dest / "sub").mkdir(parents=True)
    (dest / "a.csv").write_text("text,label\nhola,0\n")
    (dest / "sub" / "b.csv").write_text("text,label\nworld,1\n")

    entries = build_manifest_entries("daigt_test", "owner/slug", dest)

    assert len(entries) == 2
    assert all(e["dataset"] == "daigt_test" for e in entries)
    assert all(len(e["sha256"]) == 64 for e in entries)
    assert all(e["size_bytes"] > 0 for e in entries)
    # rutas en formato posix, relativas y ordenadas de forma determinista
    files = [e["file"] for e in entries]
    assert files == sorted(files)
    assert all("\\" not in f for f in files)
