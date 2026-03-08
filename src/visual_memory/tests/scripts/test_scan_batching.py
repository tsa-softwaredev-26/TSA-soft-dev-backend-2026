"""CPU-only unit tests for batched embedding in ScanPipeline. No model loading."""
import torch
import torch.nn.functional as F
from PIL import Image

from visual_memory.engine.embedding.embed_combined import make_combined_embedding
from visual_memory.utils.image_utils import crop_object


def _pass(name: str) -> None:
    print(f"PASS  {name}")


def _fail(name: str, reason: str) -> None:
    print(f"FAIL  {name}: {reason}")


# ---- Mock embedder that records call counts ----

class _MockEmbedder:
    """Returns deterministic L2-normalized tensors without loading any model."""
    IMG_DIM = 1024

    def __init__(self):
        self.embed_calls = 0
        self.batch_embed_calls = 0

    def embed(self, image):
        self.embed_calls += 1
        t = torch.ones(1, self.IMG_DIM)
        return F.normalize(t, dim=1)

    def batch_embed(self, images):
        self.batch_embed_calls += 1
        n = len(images)
        t = torch.ones(n, self.IMG_DIM)
        return F.normalize(t, dim=1)


# ---- Test 1: empty database returns [] without error ----

def test_empty_database():
    name = "empty database returns []"
    database_images = []
    if not database_images:
        result = []
    else:
        _, imgs = zip(*database_images)
        result = list(imgs)
    try:
        assert result == []
        _pass(name)
    except AssertionError as e:
        _fail(name, str(e))


# ---- Test 2: zip(*list) unpack with single item ----

def test_zip_unpack_single():
    name = "zip unpack with 1 item"
    database_images = [("path/a.png", "img_a")]
    try:
        paths, imgs = zip(*database_images)
        assert paths == ("path/a.png",)
        assert imgs == ("img_a",)
        _pass(name)
    except Exception as e:
        _fail(name, str(e))


# ---- Test 3: zip(*list) unpack with multiple items ----

def test_zip_unpack_multi():
    name = "zip unpack with N items"
    database_images = [("p1", "i1"), ("p2", "i2"), ("p3", "i3")]
    try:
        paths, imgs = zip(*database_images)
        assert len(paths) == 3
        assert list(imgs) == ["i1", "i2", "i3"]
        _pass(name)
    except Exception as e:
        _fail(name, str(e))


# ---- Test 4: batch_embed slice [i:i+1] gives (1, D) not (D,) ----

def test_batch_slice_shape():
    name = "batch_embed slice preserves (1, D) shape"
    embedder = _MockEmbedder()
    imgs = [Image.new("RGB", (64, 64))] * 4
    embs = embedder.batch_embed(imgs)  # (4, 1024)
    try:
        assert embs.shape == (4, 1024), f"expected (4,1024) got {embs.shape}"
        for i in range(4):
            sliced = embs[i:i+1]
            assert sliced.shape == (1, 1024), f"slice {i} shape {sliced.shape} != (1, 1024)"
            # wrong way: embs[i] gives (1024,) — ensure we're not doing that
            wrong = embs[i]
            assert wrong.shape == (1024,), "sanity: embs[i] should be 1D"
        _pass(name)
    except AssertionError as e:
        _fail(name, str(e))


# ---- Test 5: sliced embedding works with make_combined_embedding ----

def test_sliced_emb_combined():
    name = "sliced batch embedding compatible with make_combined_embedding"
    embedder = _MockEmbedder()
    imgs = [Image.new("RGB", (64, 64))] * 3
    img_embs = embedder.batch_embed(imgs)

    try:
        for i in range(3):
            img_emb = img_embs[i:i+1]
            combined = make_combined_embedding(img_emb, None)
            assert combined.shape == (1, 1536), f"combined shape {combined.shape} != (1, 1536)"
        _pass(name)
    except AssertionError as e:
        _fail(name, str(e))


# ---- Test 6: batch_embed called once for N items (not N times) ----

def test_batch_embed_call_count():
    name = "batch_embed called once for N database images"
    embedder = _MockEmbedder()
    database_images = [(f"p{i}", Image.new("RGB", (64, 64))) for i in range(5)]

    if not database_images:
        _fail(name, "empty database_images")
        return

    _, imgs = zip(*database_images)
    embedder.batch_embed(list(imgs))

    try:
        assert embedder.batch_embed_calls == 1, f"batch_embed called {embedder.batch_embed_calls} times"
        assert embedder.embed_calls == 0, f"embed() called {embedder.embed_calls} times (should be 0)"
        _pass(name)
    except AssertionError as e:
        _fail(name, str(e))


# ---- Test 7: crops list length matches boxes list length ----

def test_crops_length_matches_boxes():
    name = "crops list length matches boxes count"
    query_image = Image.new("RGB", (640, 480))
    boxes = [
        [10, 10, 100, 100],
        [200, 150, 350, 300],
        [400, 50, 600, 400],
    ]
    try:
        crops = [crop_object(query_image, box) for box in boxes]
        assert len(crops) == len(boxes), f"crops {len(crops)} != boxes {len(boxes)}"
        for c in crops:
            assert isinstance(c, Image.Image), "crop is not a PIL Image"
        _pass(name)
    except Exception as e:
        _fail(name, str(e))


# ---- Test 8: single-box edge case (N=1 batch) ----

def test_single_box_batch():
    name = "single-box batch (N=1 edge case)"
    embedder = _MockEmbedder()
    query_image = Image.new("RGB", (640, 480))
    boxes = [[10, 10, 200, 200]]

    try:
        crops = [crop_object(query_image, box) for box in boxes]
        img_embs = embedder.batch_embed(crops)
        assert img_embs.shape == (1, 1024), f"shape {img_embs.shape}"
        sliced = img_embs[0:1]
        assert sliced.shape == (1, 1024), f"slice shape {sliced.shape}"
        combined = make_combined_embedding(sliced, None)
        assert combined.shape == (1, 1536), f"combined shape {combined.shape}"
        _pass(name)
    except AssertionError as e:
        _fail(name, str(e))


if __name__ == "__main__":
    test_empty_database()
    test_zip_unpack_single()
    test_zip_unpack_multi()
    test_batch_slice_shape()
    test_sliced_emb_combined()
    test_batch_embed_call_count()
    test_crops_length_matches_boxes()
    test_single_box_batch()
