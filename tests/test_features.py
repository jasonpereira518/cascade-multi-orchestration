from __future__ import annotations

from cascade.taskgen.features import compute_features, split_patch
from cascade.taskgen.schema import Features, Task

GOLD = """\
diff --git a/src/pkg/core.py b/src/pkg/core.py
index 111..222 100644
--- a/src/pkg/core.py
+++ b/src/pkg/core.py
@@ -10,7 +10,8 @@ def parse_option(flag):
     x = 1
-    return flag
+    y = flag.strip()
+    return y
@@ -40,3 +41,7 @@ class Parser:
     pass
+    def new_helper(self):
+        return 1
+    def _private_helper(self):
+        return 2
diff --git a/CHANGES.md b/CHANGES.md
--- a/CHANGES.md
+++ b/CHANGES.md
@@ -1,3 +1,4 @@
+- fixed option parsing
"""

TESTS = """\
diff --git a/tests/test_core.py b/tests/test_core.py
--- a/tests/test_core.py
+++ b/tests/test_core.py
@@ -1,2 +1,5 @@
 import pkg
+def test_a():
+    assert pkg.parse_option(" -v ") == "v"
+    assert True
"""

BODY = (
    "`parse_option()` leaves whitespace in the flag.\n\n```python\nparse_option(' -v ')\n```\n\n"
    "```\nTraceback (most recent call last):\n  File x\n```\nSee also `Parser` and `os.path`.\n"
)


def features(**over: object) -> Features:
    args: dict[str, object] = {
        "issue_title": "parse_option does not strip",
        "issue_body": BODY,
        "gold_patch": GOLD,
        "test_patch": TESTS,
        "fail_to_pass": ["tests/test_core.py::test_a"],
        "pass_to_pass": ["tests/test_core.py::test_b", "tests/test_core.py::test_c"],
        "test_runtime_s": 3.5,
    }
    return compute_features(**{**args, **over})  # type: ignore[arg-type]


def test_patch_features_count_only_source_python_files() -> None:
    f = features()
    assert f.src_files == 1  # CHANGES.md is documentation, not source
    assert f.gold_files == 2 and f.test_files == 1
    assert f.src_hunks == 2
    assert (f.src_added, f.src_removed, f.src_lines) == (6, 1, 7)
    assert f.test_added == 3
    assert f.test_runtime_s == 3.5


def test_symbols_come_from_hunk_headers_and_added_or_removed_definitions() -> None:
    f = features()
    # parse_option and Parser (hunk contexts) plus new_helper and _private_helper (added defs)
    assert f.changed_symbols == 4
    assert f.new_public_symbols == 1  # new_helper; _private_helper is private


def test_a_symbol_that_is_both_added_and_removed_is_not_new() -> None:
    gold = (
        "diff --git a/src/a.py b/src/a.py\n--- a/src/a.py\n+++ b/src/a.py\n@@ -1,3 +1,3 @@\n"
        "-def rename_me(x):\n+def rename_me(x, y=0):\n     return x\n"
    )
    f = features(gold_patch=gold, test_patch="")
    assert f.new_public_symbols == 0 and f.changed_symbols == 1


def test_issue_features() -> None:
    f = features()
    assert f.issue_code_blocks == 2
    assert f.issue_has_traceback is True
    assert f.issue_identifiers == 3  # parse_option, Parser, os.path
    assert f.changed_symbols_in_issue == 2  # parse_option and Parser are named in the issue
    assert f.issue_chars == len("parse_option does not strip") + len(BODY)
    assert f.issue_words > 20


def test_symbol_matching_uses_word_boundaries() -> None:
    f = features(issue_title="x", issue_body="the Parsers and parse_options are unrelated")
    assert f.changed_symbols_in_issue == 0


def test_test_lists_are_counted() -> None:
    f = features(fail_to_pass=["a", "b", "c"], pass_to_pass=[])
    assert (f.f2p_count, f.p2p_count) == (3, 0)


def test_empty_inputs_give_zero_features_not_errors() -> None:
    f = compute_features(
        issue_title="", issue_body="", gold_patch="", test_patch="", fail_to_pass=[],
        pass_to_pass=[], test_runtime_s=0.0,
    )  # fmt: skip
    assert f == Features(src_files=0, src_lines=0, issue_chars=0, test_runtime_s=0.0)


def test_features_are_deterministic() -> None:
    assert features() == features()


def test_split_patch_returns_one_block_per_file() -> None:
    blocks = split_patch(GOLD)
    assert [p for p, _ in blocks] == ["src/pkg/core.py", "CHANGES.md"]
    assert split_patch("") == [] and split_patch("not a diff") == []


def test_old_manifest_lines_without_the_new_fields_still_load() -> None:
    old = Features.model_validate(
        {"src_files": 1, "src_lines": 2, "issue_chars": 30, "test_runtime_s": 1.0}
    )
    assert old.f2p_count == 0 and old.issue_has_traceback is False


def test_the_synthetic_task_features(task: Task) -> None:
    f = compute_features(
        issue_title=task.issue_title, issue_body=task.issue_body, gold_patch=task.gold_patch,
        test_patch=task.test_patch, fail_to_pass=task.fail_to_pass,
        pass_to_pass=task.pass_to_pass, test_runtime_s=1.0,
    )  # fmt: skip
    assert f.src_files == 1 and f.src_lines == 2 and f.test_files == 1
    assert f.changed_symbols == 1 and f.changed_symbols_in_issue == 1  # clamp is named in the issue
    assert f.f2p_count == 1 and f.p2p_count == 2
