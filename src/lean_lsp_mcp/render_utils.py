"""Extract images from ProofWidgets props.

Widgets like #png embed base64 image data in their props. This module
extracts those images for multimodal AI consumption.
"""

import re
from typing import Any, List, Tuple

# Data URL pattern for embedded images
DATA_URL_PATTERN = re.compile(
    r"data:image/(png|jpeg|jpg|gif|webp|svg\+xml);base64,([A-Za-z0-9+/=]+)"
)

# Common prop names that may contain base64 image data
IMAGE_PROP_NAMES = {"base64", "image", "png", "img", "imageData", "data", "src"}


def extract_images_from_props(props: dict) -> List[Tuple[str, str]]:
    """Extract all images from widget props.

    Handles:
    - Direct base64 props (e.g., `base64`, `image`, `png`)
    - Data URLs in string values (e.g., `data:image/png;base64,...`)
    - Nested structures (recursively searches dicts and lists)

    Args:
        props: Widget props dictionary

    Returns:
        List of (mime_type, base64_data) tuples
    """
    images: List[Tuple[str, str]] = []
    _extract_images_recursive(props, images, depth=0)
    return images


def _extract_images_recursive(
    obj: Any, images: List[Tuple[str, str]], depth: int
) -> None:
    """Recursively extract images from nested structures."""
    if depth > 10:  # Prevent infinite recursion
        return

    if isinstance(obj, str):
        # Check for data URL
        match = DATA_URL_PATTERN.search(obj)
        if match:
            mime_type = f"image/{match.group(1)}"
            b64_data = match.group(2)
            images.append((mime_type, b64_data))
        # Check if string itself looks like base64 PNG (starts with iVBOR for PNG)
        elif obj.startswith("iVBOR") and len(obj) > 100:
            images.append(("image/png", obj))
        # JPEG magic
        elif obj.startswith("/9j/") and len(obj) > 100:
            images.append(("image/jpeg", obj))

    elif isinstance(obj, dict):
        # Check known prop names first
        for prop_name in IMAGE_PROP_NAMES:
            if prop_name in obj:
                val = obj[prop_name]
                if isinstance(val, str):
                    if val.startswith("iVBOR"):
                        images.append(("image/png", val))
                    elif val.startswith("/9j/"):
                        images.append(("image/jpeg", val))
                    else:
                        # Try data URL extraction
                        match = DATA_URL_PATTERN.search(val)
                        if match:
                            images.append((f"image/{match.group(1)}", match.group(2)))

        # Recurse into all values
        for key, val in obj.items():
            if key not in IMAGE_PROP_NAMES:  # Avoid double-processing
                _extract_images_recursive(val, images, depth + 1)

    elif isinstance(obj, list):
        for item in obj:
            _extract_images_recursive(item, images, depth + 1)
