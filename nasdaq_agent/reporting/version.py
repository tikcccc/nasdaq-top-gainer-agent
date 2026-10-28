"""Renderer identity exposed by runtime config, saved reports and MIME headers."""
from __future__ import annotations

import hashlib
from pathlib import Path

REPORT_VERSION = 'evidence-v2'


def renderer_identity() -> dict[str, str]:
    root = Path(__file__).parent
    files = sorted([*root.glob('*.py'), *root.glob('templates/*.j2'),
                    root.parent / 'data' / 'primary_announcements.json'])
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root.parent).as_posix().encode())
        digest.update(path.read_bytes().replace(b'\r\n', b'\n'))
    return {'version': REPORT_VERSION, 'fingerprint': digest.hexdigest()}
