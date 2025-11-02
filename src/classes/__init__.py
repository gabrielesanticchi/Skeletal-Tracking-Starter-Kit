"""
Core classes for FIFA Skeletal Tracking dataset.

This module provides OOP interfaces for working with poses, cameras, bounding boxes,
and skeletal data.
"""

from .poses import PosesData
from .cameras import CamerasData
from .bboxes import BBoxesData
from .skeleton import Skeleton2DData, Skeleton3DData
from .metadata import ImageMetadata, VideoMetadata

__all__ = [
    'PosesData',
    'CamerasData',
    'BBoxesData',
    'Skeleton2DData',
    'Skeleton3DData',
    'ImageMetadata',
    'VideoMetadata',
]
