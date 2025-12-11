"""Render ProofWidgets HTML to images for multimodal model consumption.

This module converts ProofWidgets HTML structures (the JSON format used in
Lean's infoview) to standard HTML and renders them as PNG images.
"""

import base64
import html as html_module
import re
from typing import Optional, Any, List, Tuple

from mcp.server.fastmcp.utilities.logging import get_logger

logger = get_logger(__name__)

# Data URL pattern for extracting embedded images
DATA_URL_PATTERN = re.compile(
    r'data:image/(png|jpeg|jpg|gif|webp|svg\+xml);base64,([A-Za-z0-9+/=]+)'
)

# Common prop names that may contain base64 image data
IMAGE_PROP_NAMES = {'base64', 'image', 'png', 'img', 'imageData', 'data'}


def extract_images_from_props(props: dict) -> List[Tuple[str, str]]:
    """Extract all images from widget props using multiple strategies.

    Handles:
    - Direct base64 props (e.g., `base64`, `image`, `png`)
    - Data URLs in string values (e.g., `data:image/png;base64,...`)
    - Nested structures (recursively searches dicts and lists)

    Args:
        props: Widget props dictionary

    Returns:
        List of (mime_type, base64_data) tuples
    """
    images = []
    _extract_images_recursive(props, images, depth=0)
    return images


def _extract_images_recursive(obj: Any, images: List[Tuple[str, str]], depth: int) -> None:
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
        elif obj.startswith('iVBOR') and len(obj) > 100:
            images.append(("image/png", obj))

    elif isinstance(obj, dict):
        # Check known prop names first
        for prop_name in IMAGE_PROP_NAMES:
            if prop_name in obj:
                val = obj[prop_name]
                if isinstance(val, str):
                    if val.startswith('iVBOR') or val.startswith('/9j/'):  # PNG or JPEG magic
                        images.append(("image/png" if val.startswith('iVBOR') else "image/jpeg", val))
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


def proofwidget_to_html(pw: Any) -> str:
    """Convert ProofWidgets element structure to HTML string.

    ProofWidgets uses a specific JSON format for HTML:
      {"element": ["tag", [["attr", value], ...], [children]]}
      {"text": "string"}

    This function recursively converts that structure to standard HTML.

    Args:
        pw: ProofWidgets HTML structure (dict, str, or list)

    Returns:
        HTML string representation
    """
    if pw is None:
        return ""

    if isinstance(pw, str):
        return html_module.escape(pw)

    if isinstance(pw, list):
        return "".join(proofwidget_to_html(item) for item in pw)

    if isinstance(pw, dict):
        # Plain text node
        if "text" in pw:
            return html_module.escape(str(pw["text"]))

        # Element node: {"element": ["tag", [[attr, val], ...], [children]]}
        if "element" in pw:
            elem = pw["element"]
            if not isinstance(elem, list) or len(elem) < 3:
                return ""

            tag, attrs, children = elem[0], elem[1], elem[2]

            # Build attribute string
            attr_str = ""
            if isinstance(attrs, list):
                for attr_pair in attrs:
                    if isinstance(attr_pair, list) and len(attr_pair) >= 2:
                        attr_name, attr_val = attr_pair[0], attr_pair[1]

                        # Handle style objects specially
                        if attr_name == "style" and isinstance(attr_val, dict):
                            style_parts = []
                            for k, v in attr_val.items():
                                # Convert camelCase to kebab-case for CSS
                                css_key = _camel_to_kebab(k)
                                style_parts.append(f"{css_key}: {v}")
                            attr_str += f' style="{"; ".join(style_parts)}"'
                        else:
                            # Escape attribute value
                            escaped_val = html_module.escape(str(attr_val))
                            attr_str += f' {attr_name}="{escaped_val}"'

            # Render children
            child_html = ""
            if isinstance(children, list):
                child_html = "".join(proofwidget_to_html(c) for c in children)
            elif children:
                child_html = proofwidget_to_html(children)

            # Self-closing tags
            void_tags = {"br", "hr", "img", "input", "meta", "link", "area", "base", "col", "embed", "param", "source", "track", "wbr"}
            if tag.lower() in void_tags and not child_html:
                return f"<{tag}{attr_str} />"

            return f"<{tag}{attr_str}>{child_html}</{tag}>"

    return ""


def _camel_to_kebab(name: str) -> str:
    """Convert camelCase to kebab-case for CSS properties.

    Examples:
        backgroundColor -> background-color
        borderRadius -> border-radius
    """
    result = []
    for char in name:
        if char.isupper():
            result.append("-")
            result.append(char.lower())
        else:
            result.append(char)
    return "".join(result)


def render_widget_to_base64(
    html_structure: Any,
    background: str = "#1e1e1e",
    text_color: str = "#d4d4d4",
) -> Optional[str]:
    """Render ProofWidgets HTML to base64 PNG image.

    Args:
        html_structure: ProofWidgets HTML structure (from widget props)
        background: CSS background color (default: VS Code dark theme)
        text_color: CSS text color (default: VS Code dark theme)

    Returns:
        Base64-encoded PNG image string, or None if rendering fails
    """
    try:
        # Try html2image first (more reliable, uses browser)
        try:
            from html2image import Html2Image
            return _render_with_html2image(html_structure, background, text_color)
        except ImportError:
            pass

        # Try imgkit (requires wkhtmltoimage)
        try:
            import imgkit
            return _render_with_imgkit(html_structure, background, text_color)
        except ImportError:
            pass

        # Fallback: return None and log
        logger.warning("No HTML rendering library available. Install html2image: pip install html2image")
        return None

    except Exception as e:
        logger.warning(f"Failed to render widget to image: {e}")
        return None


def _render_with_html2image(
    html_structure: Any,
    background: str,
    text_color: str,
) -> Optional[str]:
    """Render using html2image library."""
    from html2image import Html2Image
    import tempfile
    import os

    html_body = proofwidget_to_html(html_structure)
    full_html = _wrap_html(html_body, background, text_color)

    # Create a temporary file for the output
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Use a temporary directory for output
        with tempfile.TemporaryDirectory() as tmp_dir:
            hti = Html2Image(output_path=tmp_dir)
            # Render to PNG
            filename = "widget.png"
            hti.screenshot(html_str=full_html, save_as=filename, size=(400, 100))

            output_path = os.path.join(tmp_dir, filename)
            if os.path.exists(output_path):
                with open(output_path, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return None


def _render_with_imgkit(
    html_structure: Any,
    background: str,
    text_color: str,
) -> Optional[str]:
    """Render using imgkit library (requires wkhtmltoimage)."""
    import imgkit

    html_body = proofwidget_to_html(html_structure)
    full_html = _wrap_html(html_body, background, text_color)

    options = {
        "format": "png",
        "width": 400,
        "disable-smart-width": "",
        "quiet": "",
    }

    img_bytes = imgkit.from_string(full_html, False, options=options)
    return base64.b64encode(img_bytes).decode("utf-8")


def _wrap_html(body: str, background: str, text_color: str) -> str:
    """Wrap HTML body in a full document with styling."""
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: {background};
    color: {text_color};
    padding: 8px;
    margin: 0;
    font-size: 13px;
    line-height: 1.4;
}}
code {{
    font-family: 'SF Mono', Monaco, 'Cascadia Code', Consolas, monospace;
    font-size: 12px;
}}
</style>
</head>
<body>{body}</body>
</html>"""
