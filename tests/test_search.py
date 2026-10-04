from __future__ import annotations

import re
from pathlib import Path

import pytest

from cascade.agents.search import MAX_FILE_BYTES, Bm25Index, grep, tokenize


def write(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)


REPO = {
    "src/pkg/parser.py": (
        "def parse_option(flag):\n    # handles option flags such as --verbose\n    return flag.lstrip('-')\n"
    ),
    "src/pkg/colors.py": "def style_text(text, color):\n    return f'<{color}>{text}</{color}>'\n",
    "src/pkg/__init__.py": "from pkg.parser import parse_option\n",
    "tests/test_parser.py": "from pkg import parse_option\n\n\ndef test_flag():\n    assert parse_option('-v') == 'v'\n",
    "docs/guide.rst": "Guide\n=====\n\nUse parse_option for flags.\n",
}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    write(tmp_path, REPO)
    return tmp_path


def test_tokenize_splits_camel_and_snake_but_keeps_whole_identifiers() -> None:
    assert {"parseoptionflag", "parse", "option", "flag"} <= set(tokenize("parseOptionFlag"))
    assert {"parse", "option"} <= set(tokenize("parse_option"))
    assert {"httpserver", "http", "server"} <= set(tokenize("HTTPServer"))
    assert tokenize("") == [] and tokenize("--- ###") == []


def test_search_ranks_the_relevant_file_first(repo: Path) -> None:
    hits = Bm25Index(repo).search("parse option flag", top_k=5)
    assert hits and hits[0].path == "src/pkg/parser.py"
    assert all(h.score > 0 for h in hits)
    assert not any(h.path == "src/pkg/colors.py" for h in hits[:1])


def test_search_is_deterministic(repo: Path) -> None:
    a = Bm25Index(repo).search("option flag parse", top_k=5)
    b = Bm25Index(repo).search("flag parse option", top_k=5)  # same terms, different order
    assert a == b


def test_hit_line_range_contains_the_matching_line(tmp_path: Path) -> None:
    lines = [f"x{i} = {i}" for i in range(120)]
    lines[84] = "needle_marker_zeta = True"
    write(tmp_path, {"big.py": "\n".join(lines) + "\n"})
    [hit] = Bm25Index(tmp_path).search("needle_marker_zeta", top_k=1)
    assert hit.start <= 85 <= hit.end
    assert "85: needle_marker_zeta = True" in hit.snippet or hit.end - hit.start >= 12


def test_no_match_returns_empty(repo: Path) -> None:
    assert Bm25Index(repo).search("zzzzqqqq", top_k=3) == []
    assert Bm25Index(repo).search("", top_k=3) == []


def test_at_most_two_chunks_per_file_by_default(tmp_path: Path) -> None:
    write(tmp_path, {"long.py": "\n".join(f"def target_{i}(): pass" for i in range(200)) + "\n"})
    hits = Bm25Index(tmp_path).search("target def pass", top_k=10)
    assert len(hits) == 2 and {h.path for h in hits} == {"long.py"}


def test_index_skips_hidden_cache_vcs_binary_and_oversized_files(tmp_path: Path) -> None:
    write(
        tmp_path,
        {
            "keep.py": "visible_symbol = 1\n",
            ".secret.py": "visible_symbol = 'hidden file'\n",
            ".git/config": "visible_symbol\n",
            "__pycache__/mod.py": "visible_symbol = 'cache'\n",
            "node_modules/x.py": "visible_symbol = 'deps'\n",
            "image.png": "visible_symbol",
            "huge.py": "visible_symbol = 1\n" + "# pad\n" * (MAX_FILE_BYTES // 6 + 10),
        },
    )
    paths = {h.path for h in Bm25Index(tmp_path).search("visible_symbol", top_k=10)}
    assert paths == {"keep.py"}


def test_index_ignores_symlinks(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "leak.py").write_text("symlinked_secret = 1\n")
    root = tmp_path / "root"
    root.mkdir()
    (root / "ok.py").write_text("plain = 1\n")
    (root / "link.py").symlink_to(outside / "leak.py")
    assert Bm25Index(root).search("symlinked_secret", top_k=3) == []


def test_grep_matches_with_paths_and_line_numbers(repo: Path) -> None:
    out = grep(repo, r"parse_option", "*")
    assert "src/pkg/parser.py:1: def parse_option(flag):" in out
    assert any(line.startswith("tests/test_parser.py:") for line in out)


@pytest.mark.parametrize(
    ("glob", "expected_prefix"),
    [("*.py", "src/"), ("docs/*", "docs/"), ("tests/*", "tests/")],
)
def test_grep_glob_filters_by_path(repo: Path, glob: str, expected_prefix: str) -> None:
    out = grep(repo, r"parse_option", glob)
    assert out and all(line.startswith(expected_prefix) or glob == "*.py" for line in out)
    if glob == "docs/*":
        assert all(line.startswith("docs/") for line in out)


def test_grep_caps_matches_and_rejects_bad_regex(tmp_path: Path) -> None:
    write(tmp_path, {"many.py": "hit\n" * 500})
    assert len(grep(tmp_path, "hit", "*", max_matches=7)) == 7
    with pytest.raises(re.error):
        grep(tmp_path, "([unclosed", "*")
