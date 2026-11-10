"""Test cắt chuỗi theo byte UTF-8 — Milvus đếm max_length VARCHAR bằng bytes."""
from agent.retrieval.vector import _trunc_utf8


def test_short_string_untouched():
    assert _trunc_utf8("xin chào", 100) == "xin chào"


def test_ascii_truncated_to_bytes():
    assert _trunc_utf8("abcdefgh", 5) == "abcde"


def test_vietnamese_truncated_by_bytes_not_chars():
    # "ế" = 3 bytes UTF-8 — 10 ký tự ~ 20+ bytes, cắt 10 byte thì < 10 ký tự
    s = "ếếếếế"  # 5 ký tự, 15 bytes
    out = _trunc_utf8(s, 10)
    assert len(out) < len(s)
    assert len(out.encode("utf-8")) <= 10
    assert s.startswith(out)


def test_never_splits_multi_byte_char():
    s = "ế" * 5  # 15 bytes
    out = _trunc_utf8(s, 14)  # cắt giữa byte của ký tự cuối
    assert out == "ế" * 4  # decode errors="ignore" bỏ dở dang
    assert len(out.encode("utf-8")) <= 14  # không vượt byte budget


def test_exact_fit():
    s = "ế" * 5  # 15 bytes
    assert _trunc_utf8(s, 15) == s
