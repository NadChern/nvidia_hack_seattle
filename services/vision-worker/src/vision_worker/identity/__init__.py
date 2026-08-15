"""Personal-object identity adapters and pure matching."""

from vision_worker.identity.base import (
    EmbeddingVectors,
    IdentityFrame,
    MaskedCrop,
    ObjectEmbedder,
    SegmentedDetection,
    SegmentingDetector,
)
from vision_worker.identity.gallery import GalleryCache
from vision_worker.identity.resolver import IdentityResolver, IdentityResolverConfig

__all__ = [
    "EmbeddingVectors",
    "GalleryCache",
    "IdentityFrame",
    "IdentityResolver",
    "IdentityResolverConfig",
    "MaskedCrop",
    "ObjectEmbedder",
    "SegmentedDetection",
    "SegmentingDetector",
]
