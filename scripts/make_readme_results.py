"""Regenerate the results block of README.md from artifacts/resume_stats.json.

    uv run python scripts/make_readme_results.py            # rewrite the block
    uv run python scripts/make_readme_results.py --check    # fail if the README is out of date
    uv run python scripts/make_readme_results.py --placeholder   # the "no results yet" block

Only the text between the results markers changes; the rest of the README is hand-written.
"""

from __future__ import annotations

import sys

from cascade.resume.render import readme_main

if __name__ == "__main__":
    sys.exit(readme_main())
