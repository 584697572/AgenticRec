"""Explicit parent-process key lookup; importing this module reads no secrets."""
import os
import re
from pathlib import Path

WORKSPACE_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class LocalKeyError(ValueError):
    pass


def read_api_key(path=None, environ=None):
    """Read only OPENAI_API_KEY; environment wins, no export or interpolation.

    Supports UTF-8/BOM, optional export, single/double quotes and comments.
    This deliberately does not apply general dotenv settings to the environment.
    """
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
        raise LocalKeyError("workspace .env must be readable UTF-8; contents are not logged") from None
    found = False
    for original_line in content.splitlines():
        line = original_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, value = line.partition("=")
        if name.strip() != "OPENAI_API_KEY":
            if re.match(r"^OPENAI_API_KEY(?:\s|$)", line):
                raise LocalKeyError("invalid OPENAI_API_KEY assignment; contents are not logged")
            continue
        if not separator or found:
            raise LocalKeyError("OPENAI_API_KEY must have exactly one assignment; contents are not logged")
        found = True
        value = value.strip()
        if value[:1] in ("'", '"'):
            quote = value[0]
            end = value.find(quote, 1)
            if end < 0 or (value[end + 1:].strip() and not value[end + 1:].lstrip().startswith("#")):
                raise LocalKeyError("invalid quoted OPENAI_API_KEY; contents are not logged")
            value = value[1:end]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
            if value.startswith("#"):
                value = ""
        if any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in value):
            raise LocalKeyError("OPENAI_API_KEY must be a single value; contents are not logged")
        key = value
    return key
