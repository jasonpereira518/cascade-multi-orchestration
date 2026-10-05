"""Fill docs/WRITEUP.template.md from artifacts/resume_stats.json and write docs/WRITEUP.md.

uv run python scripts/make_writeup.py            # write docs/WRITEUP.md
uv run python scripts/make_writeup.py --check    # fail if docs/WRITEUP.md is out of date
"""

from __future__ import annotations

import sys

from cascade.resume.render import writeup_main

if __name__ == "__main__":
    sys.exit(writeup_main())
