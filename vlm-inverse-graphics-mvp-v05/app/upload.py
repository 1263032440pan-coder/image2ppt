"""Lightweight safe upload normalization (no CV/skimage imports in API server)."""
from __future__ import annotations
from io import BytesIO
from PIL import Image


def to_jpeg_bytes(raw: bytes, max_side: int=1280) -> bytes:
    if len(raw)>10*1024*1024:
        raise ValueError('Image too large. Max 10 MB.')
    try:
        with Image.open(BytesIO(raw)) as img:
            if img.width*img.height>25_000_000:
                raise ValueError('Too many pixels; 25 million maximum.')
            img=img.convert('RGB')
            img.thumbnail((max_side,max_side),Image.Resampling.LANCZOS)
            output=BytesIO()
            img.save(output,format='JPEG',quality=86,optimize=True)
            return output.getvalue()
    except (OSError,Image.DecompressionBombError) as exc:
        raise ValueError('Invalid or unsupported image.') from exc
