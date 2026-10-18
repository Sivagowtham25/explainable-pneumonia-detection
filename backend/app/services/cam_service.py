"""
Grad-CAM image serialization, encoding, and conversion service.
"""

import io
import base64
from typing import Optional
from PIL import Image


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    """
    Parses raw binary image bytes into a verified PIL RGB Image.

    Parameters:
    -----------
    image_bytes : bytes
        Raw binary data from uploaded file.

    Returns:
    --------
    PIL.Image.Image
        Verified RGB Image.

    Raises:
    -------
    ValueError
        If bytes cannot be identified as a valid image.
    """
    try:
        stream = io.BytesIO(image_bytes)
        img = Image.open(stream)
        img.verify()  # Verify image integrity
        stream.seek(0)
        return Image.open(stream).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Uploaded file is not a valid or readable image: {str(exc)}") from exc


def image_to_base64(img: Image.Image, format: str = "PNG") -> str:
    """
    Converts a PIL Image to a base64 Data URI string suitable for browser display.

    Parameters:
    -----------
    img : PIL.Image.Image
        The image to encode.
    format : str
        Target image encoding format ('PNG', 'JPEG', etc.).

    Returns:
    --------
    str
        Formatted Data URI (e.g. 'data:image/png;base64,...').
    """
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime_type = "image/png" if format.upper() == "PNG" else "image/jpeg"
    return f"data:{mime_type};base64,{encoded}"
