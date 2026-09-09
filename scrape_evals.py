"""
Horn of Africa Inspect Evals

Local preprocessing engine for multilingual code-switching evaluation data.

The parser is designed to prepare authorized text datasets containing
Amharic, Oromo, Somali, and English code-switching for downstream
evaluation with Inspect AI.

It performs Unicode normalization, generic log-noise removal,
tokenization, script detection, and basic code-switch detection.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterator


# Ethiopic Unicode ranges used by Amharic, Oromo, Tigrinya,
# and other languages written with Ethiopic script.
ETHIOPIC_RE = re.compile(
    r"[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\uAB01-\uAB2F]"
)

# Latin-script words, including common apostrophe/hyphen forms.
LATIN_RE = re.compile(
    r"\b[A-Za-z]+(?:['’-][A-Za-z]+)*\b"
)

# Tokenization pattern for mixed-script text.
TOKEN_RE = re.compile(
    r"[A-Za-z]+(?:['’-][A-Za-z]+)*"
    r"|[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\uAB01-\uAB2F]+"
    r"|\d+(?:[.,]\d+)*"
    r"|[^\s]"
)

# Generic system/log prefixes.
# These are deliberately platform-independent.
NOISE_PATTERNS = [
    re.compile(
        r"^\s*(?:bot|system|automated message)\s*:\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*\[[0-9]{1,2}:[0-9]{2}"
        r"(?::[0-9]{2})?\]\s*"
    ),
]


def normalize_unicode(text: str) -> str:
    """Normalize Unicode without changing the linguistic content."""
    text = unicodedata.normalize("NFC", text)

    # Remove invisible formatting characters that can interfere
    # with downstream tokenization.
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")

    # Normalize line endings.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    return text


def strip_noise(text: str) -> str:
    """Remove generic bot/log prefixes from each line."""
    cleaned_lines = []

    for line in text.splitlines():
        cleaned = line

        for pattern in NOISE_PATTERNS:
            cleaned = pattern.sub("", cleaned)

        cleaned = cleaned.strip()

        if cleaned:
            cleaned_lines.append(cleaned)

    return "\n".join(cleaned_lines)


def tokenize(text: str) -> list[str]:
    """Tokenize mixed Ethiopic/Latin text."""
    return TOKEN_RE.findall(text)


def detect_scripts(text: str) -> list[str]:
    """
    Detect the writing systems present.

    Ethiopic script is shared by multiple languages, so this function
    reports the script rather than incorrectly claiming exact language ID.
    """
    scripts = []

    if ETHIOPIC_RE.search(text):
        scripts.append("ethiopic-script")

    if LATIN_RE.search(text):
        scripts.append("latin-script")

    if not scripts:
        scripts.append("other")

    return scripts


def detect_code_switching(text: str) -> bool:
    """Return True when Ethiopic and Latin scripts occur together."""
    return bool(
        ETHIOPIC_RE.search(text)
        and LATIN_RE.search(text)
    )


def parse_record(text: str, record_id: int) -> dict:
    """Convert one raw text item into a structured evaluation record."""

    normalized = normalize_unicode(text)
    normalized = strip_noise(normalized)

    tokens = tokenize(normalized)

    return {
        "id": record_id,
        "raw_text": text,
        "normalized_text": normalized,
        "scripts": detect_scripts(normalized),
        "code_switched": detect_code_switching(normalized),
        "tokens": tokens,
        "token_count": len(tokens),
        "character_count": len(normalized),
    }


def read_txt(path: Path) -> Iterator[str]:
    """Read one text record per non-empty line."""
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                yield line


def read_jsonl(path: Path) -> Iterator[str]:
    """
    Read text records from JSONL.

    Supported fields:
        text
        message
        content
    """
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

            if isinstance(record, str):
                yield record
                continue

            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number} must contain "
                    "a JSON object or string."
                )

            for field in ("text", "message", "content"):
                value = record.get(field)

                if isinstance(value, str):
                    yield value
                    break
            else:
                raise ValueError(
                    f"Line {line_number} does not contain "
                    "'text', 'message', or 'content'."
                )


def read_input(path: Path) -> Iterator[str]:
    """Select the appropriate reader based on file extension."""

    if path.suffix.lower() == ".txt":
        yield from read_txt(path)

    elif path.suffix.lower() == ".jsonl":
        yield from read_jsonl(path)

    else:
        raise ValueError(
            "Unsupported file format. "
            "Use .txt or .jsonl."
        )


def write_jsonl(
    records: Iterator[dict],
    output_path: Path,
) -> int:
    """Write parsed records to JSONL."""
    count = 0

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare mixed Ethiopic/English text "
            "for multilingual evaluation."
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input .txt or .jsonl file.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("parsed.jsonl"),
        help="Output JSONL file.",
    )

    args = parser.parse_args()

    if not args.input.exists():
        parser.error(
            f"Input file does not exist: {args.input}"
        )

    if not args.input.is_file():
        parser.error(
            f"Input path is not a file: {args.input}"
        )

    records = (
        parse_record(text, record_id)
        for record_id, text in enumerate(
            read_input(args.input),
            start=1,
        )
    )

    count = write_jsonl(
        records,
        args.output,
    )

    print(
        f"Successfully parsed {count} record(s)."
    )

    print(
        f"Output written to: {args.output}"
    )


if __name__ == "__main__":
    main()
