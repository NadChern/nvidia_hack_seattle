"""C-RADIO adapter smoke test; opt in with `-m models`."""

from __future__ import annotations

import numpy as np
import pytest

from vision_worker.identity.base import MaskedCrop
from vision_worker.identity.radio import RadioEmbedder

pytestmark = [pytest.mark.anyio, pytest.mark.models]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_radio_returns_matching_normalized_summary_and_spatial_vectors() -> None:
    image = np.full((512, 512, 3), 127, dtype=np.uint8)
    image[96:416, 128:384] = (180, 40, 20)
    mask = np.zeros((512, 512), dtype=np.bool_)
    mask[96:416, 128:384] = True
    embedder = RadioEmbedder(device="cuda")
    await embedder.initialize()

    [vectors] = await embedder.embed((MaskedCrop(image=image, mask=mask),))

    assert vectors.dim > 0
    assert vectors.summary.shape == vectors.pooled_spatial.shape
    assert np.linalg.norm(vectors.summary) == pytest.approx(1.0, abs=1e-4)
    assert np.linalg.norm(vectors.pooled_spatial) == pytest.approx(1.0, abs=1e-4)
    await embedder.aclose()
