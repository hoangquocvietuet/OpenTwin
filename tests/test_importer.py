import json
import os
import tempfile
import zipfile

import pytest

from app.importer import (
    validate_zip,
    find_inbox_folder,
    ZipValidationError,
)


def _create_test_zip(tmp_path, with_inbox=True) -> str:
    """Create a minimal test zip with Facebook-like structure."""
    zip_path = str(tmp_path / "test_export.zip")

    with zipfile.ZipFile(zip_path, "w") as zf:
        if with_inbox:
            conv_data = {
                "participants": [
                    {"name": "Hoàng Quốc Việt"},
                    {"name": "Friend"},
                ],
                "messages": [
                    {
                        "sender_name": "Friend",
                        "timestamp_ms": 1690000000000,
                        "content": "xin chào",
                    },
                    {
                        "sender_name": "Hoàng Quốc Việt",
                        "timestamp_ms": 1690000060000,
                        "content": "chào bạn",
                    },
                ],
            }
            zf.writestr(
                "inbox/friend_123/message_1.json",
                json.dumps(conv_data, ensure_ascii=False),
            )
    return zip_path


def test_validate_zip_accepts_valid(tmp_path):
    """Valid zip file passes validation."""
    zip_path = _create_test_zip(tmp_path)
    assert validate_zip(zip_path) is True


def test_validate_zip_rejects_non_zip(tmp_path):
    """Non-zip file raises ZipValidationError."""
    bad_path = str(tmp_path / "not_a_zip.txt")
    with open(bad_path, "w") as f:
        f.write("not a zip")
    with pytest.raises(ZipValidationError, match="zip"):
        validate_zip(bad_path)


def test_validate_zip_rejects_oversized(tmp_path):
    """Zip over 500MB raises ZipValidationError."""
    zip_path = _create_test_zip(tmp_path)
    with pytest.raises(ZipValidationError, match="large"):
        validate_zip(zip_path, max_size_mb=0)  # 0 MB limit = always too large


def test_find_inbox_folder_direct(tmp_path):
    """Finds inbox/ folder at top level."""
    inbox_dir = tmp_path / "inbox" / "friend_123"
    inbox_dir.mkdir(parents=True)
    (inbox_dir / "message_1.json").write_text("{}")

    result = find_inbox_folder(str(tmp_path))
    assert result.endswith("inbox")


def test_find_inbox_folder_nested(tmp_path):
    """Finds inbox/ folder nested one or two levels deep."""
    inbox_dir = tmp_path / "your_facebook_activity" / "messages" / "inbox" / "friend_123"
    inbox_dir.mkdir(parents=True)
    (inbox_dir / "message_1.json").write_text("{}")

    result = find_inbox_folder(str(tmp_path))
    assert result.endswith("inbox")


def test_find_inbox_folder_missing(tmp_path):
    """Returns None when no inbox/ folder exists."""
    result = find_inbox_folder(str(tmp_path))
    assert result is None
