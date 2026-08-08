"""Regression tests for `/ingest` path-traversal and validation hardening.

The endpoint previously built the destination path as
`Path("data/uploads") / file.filename` with no sanitization at all, so a
crafted multipart `filename` (`../../etc/passwd`, an absolute path, a
symlink already sitting in the upload directory, ...) could write
outside `data/uploads/`. `main._resolve_safe_upload_path` now strips the
client-supplied filename down to its basename before ever building a
path, then re-resolves and verifies containment under the upload root --
which also catches a symlink escape that basename-stripping alone can't
see. These tests exercise that function directly (fast, no document
processing needed) and the full `/ingest` endpoint end-to-end for both
an attack and a legitimate upload.
"""

from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient


@pytest.fixture
def sandboxed_upload_root(app_module, tmp_path, monkeypatch):
    """Point `main._UPLOAD_ROOT` at a throwaway directory for the
    duration of one test, so these tests never touch the real
    `data/uploads/` in the repo.
    """

    upload_root = tmp_path / "uploads"
    monkeypatch.setattr(app_module, "_UPLOAD_ROOT", upload_root)
    return upload_root


# --- `_resolve_safe_upload_path` unit tests ---------------------------------


def test_resolve_safe_upload_path_accepts_plain_filename(app_module, sandboxed_upload_root):
    dest = app_module._resolve_safe_upload_path("report.pdf")
    assert dest == (sandboxed_upload_root / "report.pdf").resolve()
    assert dest.parent == sandboxed_upload_root.resolve()


@pytest.mark.parametrize(
    "malicious_name",
    [
        "../../../../etc/passwd",
        "..\\..\\..\\windows\\win.ini",
        "/etc/passwd",
        "C:\\Windows\\System32\\config\\SAM",
        "....//....//etc/passwd",
        "a/../../b.txt",
        "../../../secrets.env",
    ],
)
def test_resolve_safe_upload_path_confines_traversal_attempts_to_upload_root(
    app_module, sandboxed_upload_root, malicious_name
):
    """A traversal-shaped filename must never resolve outside the upload
    root -- it's normalized down to its basename and safely contained,
    not merely rejected with `..` intact somewhere.
    """

    dest = app_module._resolve_safe_upload_path(malicious_name)
    upload_root = sandboxed_upload_root.resolve()

    assert dest.parent == upload_root
    assert upload_root in dest.parents


@pytest.mark.parametrize(
    "bad_name", ["", None, ".", "..", "evil.txt:hidden.exe", "con.txt", "NUL"]
)
def test_resolve_safe_upload_path_rejects_invalid_filenames(app_module, sandboxed_upload_root, bad_name):
    with pytest.raises(HTTPException) as exc_info:
        app_module._resolve_safe_upload_path(bad_name)
    assert exc_info.value.status_code == 400


def test_resolve_safe_upload_path_rejects_null_byte():
    from main import _resolve_safe_upload_path

    with pytest.raises(HTTPException) as exc_info:
        _resolve_safe_upload_path("evil\x00.txt")
    assert exc_info.value.status_code == 400


def test_resolve_safe_upload_path_rejects_symlink_escape(app_module, sandboxed_upload_root):
    """A same-named entry already sitting in the upload directory that is
    actually a symlink to somewhere outside it must be rejected --
    basename-stripping alone can't catch this, since the traversal
    happens on disk, not in the filename string.
    """

    sandboxed_upload_root.mkdir(parents=True, exist_ok=True)
    outside_dir = sandboxed_upload_root.parent / "outside"
    outside_dir.mkdir()
    secret = outside_dir / "secret.txt"
    secret.write_text("do not read me")

    link = sandboxed_upload_root / "innocuous.txt"
    try:
        link.symlink_to(secret)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted in this environment")

    with pytest.raises(HTTPException) as exc_info:
        app_module._resolve_safe_upload_path("innocuous.txt")
    assert exc_info.value.status_code == 400


# --- full `/ingest` endpoint tests -------------------------------------------


def test_ingest_endpoint_confines_traversal_filename_inside_upload_root(app_module, sandboxed_upload_root):
    client = TestClient(app_module.app)
    response = client.post(
        "/ingest",
        files={"file": ("../../../evil.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 200
    upload_root = sandboxed_upload_root.resolve()
    written = list(upload_root.iterdir())
    assert len(written) == 1
    assert written[0].parent == upload_root
    # must not have escaped to the upload root's parent (or further up),
    # which is exactly where "../../../evil.txt" would have landed
    # without basename-stripping.
    assert not (upload_root.parent / "evil.txt").exists()
    assert not (upload_root.parent.parent / "evil.txt").exists()


def test_ingest_endpoint_rejects_symlink_escape(app_module, sandboxed_upload_root):
    sandboxed_upload_root.mkdir(parents=True, exist_ok=True)
    outside_dir = sandboxed_upload_root.parent / "outside"
    outside_dir.mkdir()
    secret = outside_dir / "secret.txt"
    secret.write_text("do not read me")

    link = sandboxed_upload_root / "innocuous.txt"
    try:
        link.symlink_to(secret)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted in this environment")

    client = TestClient(app_module.app)
    response = client.post(
        "/ingest",
        files={"file": ("innocuous.txt", b"attacker-controlled overwrite", "text/plain")},
    )

    assert response.status_code == 400
    assert secret.read_text() == "do not read me"


def test_ingest_endpoint_accepts_legitimate_upload(app_module, sandboxed_upload_root, monkeypatch):
    monkeypatch.setattr(
        app_module,
        "_document_processor",
        type("StubProcessor", (), {"process": staticmethod(lambda path: (None, []))})(),
    )
    monkeypatch.setattr(app_module._vector_store, "add_chunks", lambda chunks: 0)
    monkeypatch.setattr(app_module._retriever, "index_corpus", lambda chunks: None)

    client = TestClient(app_module.app)
    response = client.post(
        "/ingest",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["file"] == "notes.txt"
    assert (sandboxed_upload_root / "notes.txt").read_bytes() == b"hello world"


def test_ingest_endpoint_returns_clear_error_when_ocr_unavailable(app_module, sandboxed_upload_root, monkeypatch):
    """A document that genuinely needs OCR, on a server with no usable
    Tesseract/Poppler install, must come back as a clear 422 -- not an
    unhandled 500 from a raw pdf2image/pytesseract exception.
    """

    from src.parsers.ocr_parser import OCRDependencyError

    def _raise(path):
        raise OCRDependencyError("tesseract binary not found on PATH")

    monkeypatch.setattr(
        app_module,
        "_document_processor",
        type("StubProcessor", (), {"process": staticmethod(_raise)})(),
    )

    client = TestClient(app_module.app)
    response = client.post(
        "/ingest",
        files={"file": ("scan.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )

    assert response.status_code == 422
    assert "OCR" in response.json()["detail"]
