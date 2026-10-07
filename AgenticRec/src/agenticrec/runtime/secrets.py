"""Narrow credential lookup; importing this module never reads a secret."""

import os
from pathlib import Path
import re


WORKSPACE_ENV_FILE = Path(__file__).resolve().parents[4] / ".env"


class LocalKeyError(ValueError):
    """A local key file is malformed; messages never include its contents."""


def read_api_key(path=None, environ=None):
    """Read only OPENAI_API_KEY without exporting or expanding other values."""
    environ = os.environ if environ is None else environ
    key = environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    path = WORKSPACE_ENV_FILE if path is None else Path(path)
    try:
        content = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return ""
    except (OSError, UnicodeError):
        raise LocalKeyError(
            "workspace .env must be readable UTF-8; contents are not logged"
        ) from None

    found = False
    result = ""
    for original_line in content.splitlines():
        line = original_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, value = line.partition("=")
        if name.strip() != "OPENAI_API_KEY":
            if re.match(r"^OPENAI_API_KEY(?:\s|$)", line):
                raise LocalKeyError(
                    "invalid OPENAI_API_KEY assignment; contents are not logged"
                )
            continue
        if not separator or found:
            raise LocalKeyError(
                "OPENAI_API_KEY must have exactly one assignment; contents are not logged"
            )
        found = True
        value = value.strip()
        if value[:1] in ("'", '"'):
            quote = value[0]
            end = value.find(quote, 1)
            trailing = "" if end < 0 else value[end + 1:].strip()
            if end < 0 or (trailing and not trailing.startswith("#")):
                raise LocalKeyError(
                    "invalid quoted OPENAI_API_KEY; contents are not logged"
                )
            value = value[1:end]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
            if value.startswith("#"):
                value = ""
        if any(
            character.isspace() or ord(character) < 32 or ord(character) == 127
            for character in value
        ):
            raise LocalKeyError(
                "OPENAI_API_KEY must be a single value; contents are not logged"
            )
        result = value
    return result
