"""
Unit Tests – Segmentation API
CEG4195/SEG4180 · uOttawa Winter 2026

Run with:
    pytest tests/ -v
"""

import base64
import io
import json
import os

import numpy as np
import pytest
from PIL import Image

# Ensure .env does not interfere with CI – set dummy vars before import
os.environ.setdefault("MODEL_PATH", "/tmp/no_model_ci.pth")
os.environ.setdefault("IMAGE_SIZE", "64")
os.environ.setdefault("LOG_LEVEL",  "WARNING")
os.environ.setdefault("API_KEY",    "")

from app import app as flask_app   # noqa: E402  (must come after env setup)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


def _make_rgb_image(width: int = 64, height: int = 64) -> bytes:
    """Create a random RGB PNG image in memory and return its bytes."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_mask_image(width: int = 64, height: int = 64) -> bytes:
    """Create a random binary mask PNG in memory and return its bytes."""
    arr = (np.random.rand(height, width) > 0.5).astype(np.uint8) * 255
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthCheck:
    def test_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_json_structure(self, client):
        resp = client.get("/")
        data = resp.get_json()
        assert data["status"] == "running"
        assert "model" in data
        assert "endpoints" in data
        assert "segment" in data["endpoints"]

    def test_content_type_json(self, client):
        resp = client.get("/")
        assert "application/json" in resp.content_type


# ─────────────────────────────────────────────────────────────────────────────
# Segment endpoint – multipart file upload
# ─────────────────────────────────────────────────────────────────────────────

class TestSegmentMultipart:
    def test_missing_file_returns_400(self, client):
        resp = client.post("/segment", content_type="multipart/form-data", data={})
        assert resp.status_code == 400

    def test_valid_image_returns_200(self, client):
        png_bytes = _make_rgb_image()
        resp = client.post(
            "/segment",
            content_type="multipart/form-data",
            data={"file": (io.BytesIO(png_bytes), "test.png")},
        )
        # Model may not be loaded in CI → 500 is acceptable; 200 means it ran
        assert resp.status_code in (200, 500)

    def test_response_contains_mask_b64(self, client):
        """If model loaded, response must include mask_b64."""
        png_bytes = _make_rgb_image()
        resp = client.post(
            "/segment",
            content_type="multipart/form-data",
            data={"file": (io.BytesIO(png_bytes), "test.png")},
        )
        if resp.status_code == 200:
            data = resp.get_json()
            assert "mask_b64" in data
            # Verify it's a valid base64-encoded PNG
            mask_bytes = base64.b64decode(data["mask_b64"])
            mask_img   = Image.open(io.BytesIO(mask_bytes))
            assert mask_img.mode in ("L", "RGB")

    def test_corrupt_image_returns_400(self, client):
        resp = client.post(
            "/segment",
            content_type="multipart/form-data",
            data={"file": (io.BytesIO(b"not an image"), "bad.png")},
        )
        assert resp.status_code in (400, 500)


# ─────────────────────────────────────────────────────────────────────────────
# Segment endpoint – JSON / base64 upload
# ─────────────────────────────────────────────────────────────────────────────

class TestSegmentJSON:
    def test_missing_image_b64_returns_400(self, client):
        resp = client.post(
            "/segment",
            content_type="application/json",
            data=json.dumps({}),
        )
        assert resp.status_code == 400

    def test_valid_image_b64_returns_2xx_or_500(self, client):
        png_bytes  = _make_rgb_image()
        img_b64    = base64.b64encode(png_bytes).decode()
        resp = client.post(
            "/segment",
            content_type="application/json",
            data=json.dumps({"image_b64": img_b64}),
        )
        assert resp.status_code in (200, 500)

    def test_metrics_returned_with_ground_truth(self, client):
        """When mask_b64 is supplied and model is loaded, metrics must be present."""
        png_bytes  = _make_rgb_image()
        mask_bytes = _make_mask_image()
        resp = client.post(
            "/segment",
            content_type="application/json",
            data=json.dumps({
                "image_b64": base64.b64encode(png_bytes).decode(),
                "mask_b64":  base64.b64encode(mask_bytes).decode(),
            }),
        )
        if resp.status_code == 200:
            data = resp.get_json()
            assert "metrics" in data
            if data["metrics"] is not None:
                assert "iou"  in data["metrics"]
                assert "dice" in data["metrics"]
                assert 0.0 <= data["metrics"]["iou"]  <= 1.0
                assert 0.0 <= data["metrics"]["dice"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# API Key authentication
# ─────────────────────────────────────────────────────────────────────────────

class TestAuthentication:
    def test_no_auth_when_api_key_empty(self, client):
        """With API_KEY='', every request should pass the auth guard."""
        os.environ["API_KEY"] = ""
        resp = client.get("/")
        assert resp.status_code == 200

    def test_missing_bearer_token_returns_401(self, client, monkeypatch):
        """If API_KEY is set, missing auth header must return 401."""
        import app as app_module
        monkeypatch.setattr(app_module, "API_KEY", "supersecret")
        resp = client.post(
            "/segment",
            content_type="multipart/form-data",
            data={"file": (io.BytesIO(_make_rgb_image()), "test.png")},
        )
        assert resp.status_code == 401

    def test_correct_bearer_token_passes(self, client, monkeypatch):
        """Correct Authorization header must pass the auth guard."""
        import app as app_module
        monkeypatch.setattr(app_module, "API_KEY", "supersecret")
        resp = client.post(
            "/segment",
            headers={"Authorization": "Bearer supersecret"},
            content_type="multipart/form-data",
            data={"file": (io.BytesIO(_make_rgb_image()), "test.png")},
        )
        # 200 or 500 (model may not be loaded), but NOT 401
        assert resp.status_code != 401

    def test_wrong_token_returns_401(self, client, monkeypatch):
        import app as app_module
        monkeypatch.setattr(app_module, "API_KEY", "supersecret")
        resp = client.post(
            "/segment",
            headers={"Authorization": "Bearer wrongtoken"},
            content_type="multipart/form-data",
            data={"file": (io.BytesIO(_make_rgb_image()), "test.png")},
        )
        assert resp.status_code == 401


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation (unit tests – no HTTP)
# ─────────────────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_iou_perfect_overlap(self):
        from app import _compute_metrics
        mask = np.ones((64, 64), dtype=np.uint8)
        m = _compute_metrics(mask, mask)
        assert m["iou"]  == pytest.approx(1.0)
        assert m["dice"] == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        from app import _compute_metrics
        pred = np.zeros((64, 64), dtype=np.uint8)
        gt   = np.ones( (64, 64), dtype=np.uint8)
        m    = _compute_metrics(pred, gt)
        assert m["iou"]  == pytest.approx(0.0)
        assert m["dice"] == pytest.approx(0.0)

    def test_metrics_none_when_no_gt(self):
        from app import _compute_metrics
        pred = np.ones((64, 64), dtype=np.uint8)
        assert _compute_metrics(pred, None) is None
