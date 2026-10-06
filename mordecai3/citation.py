"""A one-time reminder to cite Mordecai, shown the first time a Geoparser is built.

Printed once per machine, to stderr so it never mixes into a program's output.
Silenced by MORDECAI_NO_CITATION_NOTICE=1 or under CI. If the marker recording
that it was shown cannot be written, it is not shown at all, so a read-only
home directory never turns it into a message on every run.
"""

import os
import sys
from pathlib import Path

NOTICE = """\
Thanks for using Mordecai! If you use it in research, please cite:

  Halterman, Andrew. "Mordecai 3: A Neural Geoparser and Event Geocoder."
  arXiv preprint arXiv:2303.13675 (2023).

`mordecai3 cite` prints the BibTeX. This message is shown only once.
"""


def _marker():
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "mordecai3" / "citation_notice_shown"


def maybe_show_citation_notice():
    if os.environ.get("MORDECAI_NO_CITATION_NOTICE") or os.environ.get("CI"):
        return
    try:
        marker = _marker()
        if marker.exists():
            return
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    except OSError:
        return
    print(NOTICE, file=sys.stderr)
