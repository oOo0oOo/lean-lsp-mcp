"""Extract and render images from ProofWidgets props.

Widgets like #png embed base64 image data in their props. React components
like Recharts need headless browser rendering. This module handles both.
"""

import base64
import html as html_module
import re
import tempfile
import os
from typing import Any, List, Optional, Tuple

from mcp.server.fastmcp.utilities.logging import get_logger

logger = get_logger(__name__)

# Data URL pattern for embedded images
IMAGE_DATA_URL_PATTERN = re.compile(
    r"data:image/(png|jpeg|jpg|gif|webp|svg\+xml);base64,([A-Za-z0-9+/=]+)"
)

# Data URL pattern for embedded audio
AUDIO_DATA_URL_PATTERN = re.compile(
    r"data:audio/(wav|mp3|mpeg|ogg|flac|aac|mp4);base64,([A-Za-z0-9+/=]+)"
)

# Common prop names that may contain base64 image data
IMAGE_PROP_NAMES = {"base64", "image", "png", "img", "imageData", "data", "src"}

# Common prop names that may contain base64 audio data
AUDIO_PROP_NAMES = {"audio", "audioData", "sound", "wav", "mp3"}


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


def extract_audio_from_props(props: dict) -> List[Tuple[str, str]]:
    """Extract all audio from widget props.

    Handles:
    - Direct base64 props (e.g., `audio`, `sound`, `wav`)
    - Data URLs in string values (e.g., `data:audio/wav;base64,...`)
    - Nested structures (recursively searches dicts and lists)

    Args:
        props: Widget props dictionary

    Returns:
        List of (mime_type, base64_data) tuples
    """
    audio: List[Tuple[str, str]] = []
    _extract_media_recursive(props, [], audio, depth=0)
    return audio


def _extract_media_recursive(
    obj: Any, images: List[Tuple[str, str]], audio: List[Tuple[str, str]], depth: int
) -> None:
    """Recursively extract images and audio from nested structures."""
    if depth > 10:  # Prevent infinite recursion
        return

    if isinstance(obj, str):
        # Check for image data URL
        match = IMAGE_DATA_URL_PATTERN.search(obj)
        if match:
            mime_type = f"image/{match.group(1)}"
            b64_data = match.group(2)
            images.append((mime_type, b64_data))
        # Check for audio data URL
        match = AUDIO_DATA_URL_PATTERN.search(obj)
        if match:
            mime_type = f"audio/{match.group(1)}"
            b64_data = match.group(2)
            audio.append((mime_type, b64_data))
        # Check if string itself looks like base64 PNG (starts with iVBOR for PNG)
        elif obj.startswith("iVBOR") and len(obj) > 100:
            images.append(("image/png", obj))
        # JPEG magic
        elif obj.startswith("/9j/") and len(obj) > 100:
            images.append(("image/jpeg", obj))
        # WAV magic (UklGR = "RIFF" in base64)
        elif obj.startswith("UklGR") and len(obj) > 100:
            audio.append(("audio/wav", obj))

    elif isinstance(obj, dict):
        # Check known image prop names
        for prop_name in IMAGE_PROP_NAMES:
            if prop_name in obj:
                val = obj[prop_name]
                if isinstance(val, str):
                    if val.startswith("iVBOR"):
                        images.append(("image/png", val))
                    elif val.startswith("/9j/"):
                        images.append(("image/jpeg", val))
                    else:
                        match = IMAGE_DATA_URL_PATTERN.search(val)
                        if match:
                            images.append((f"image/{match.group(1)}", match.group(2)))

        # Check known audio prop names
        for prop_name in AUDIO_PROP_NAMES:
            if prop_name in obj:
                val = obj[prop_name]
                if isinstance(val, str):
                    if val.startswith("UklGR"):
                        audio.append(("audio/wav", val))
                    else:
                        match = AUDIO_DATA_URL_PATTERN.search(val)
                        if match:
                            audio.append((f"audio/{match.group(1)}", match.group(2)))

        # Recurse into all values
        skip_props = IMAGE_PROP_NAMES | AUDIO_PROP_NAMES
        for key, val in obj.items():
            if key not in skip_props:
                _extract_media_recursive(val, images, audio, depth + 1)

    elif isinstance(obj, list):
        for item in obj:
            _extract_media_recursive(item, images, audio, depth + 1)


def _extract_images_recursive(
    obj: Any, images: List[Tuple[str, str]], depth: int
) -> None:
    """Recursively extract images from nested structures (legacy wrapper)."""
    _extract_media_recursive(obj, images, [], depth)


def _camel_to_kebab(name: str) -> str:
    """Convert camelCase to kebab-case for CSS properties."""
    result = []
    for char in name:
        if char.isupper():
            result.append("-")
            result.append(char.lower())
        else:
            result.append(char)
    return "".join(result)


def proofwidget_to_html(pw: Any) -> str:
    """Convert ProofWidgets element structure to HTML string.

    ProofWidgets uses a specific JSON format for HTML:
      {"element": ["tag", [["attr", value], ...], [children]]}
      {"text": "string"}
      {"component": ["hash", "ComponentName", props, children]}

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
                                css_key = _camel_to_kebab(k)
                                style_parts.append(f"{css_key}: {v}")
                            attr_str += f' style="{"; ".join(style_parts)}"'
                        else:
                            escaped_val = html_module.escape(str(attr_val))
                            attr_str += f' {attr_name}="{escaped_val}"'

            # Render children
            child_html = ""
            if isinstance(children, list):
                child_html = "".join(proofwidget_to_html(c) for c in children)
            elif children:
                child_html = proofwidget_to_html(children)

            # Self-closing tags
            void_tags = {
                "br", "hr", "img", "input", "meta", "link",
                "area", "base", "col", "embed", "param", "source", "track", "wbr"
            }
            if tag.lower() in void_tags and not child_html:
                return f"<{tag}{attr_str} />"

            return f"<{tag}{attr_str}>{child_html}</{tag}>"

        # React component node: {"component": ["hash", "Name", props, children]}
        # These need JS rendering - return placeholder
        if "component" in pw:
            comp = pw["component"]
            if isinstance(comp, list) and len(comp) >= 2:
                comp_name = comp[1] if len(comp) > 1 else "Component"
                return f'<div class="react-component" data-component="{comp_name}">[React: {comp_name}]</div>'

    return ""


def render_widget_to_png(
    widget_props: dict,
    width: int = 600,
    height: int = 400,
) -> Optional[str]:
    """Render a ProofWidget to PNG using headless browser.

    Converts Recharts widget data to Chart.js and renders via headless browser.
    Chart.js is used because it loads reliably from CDN (unlike Recharts which
    has React dependency issues in headless mode).

    For lighter-weight rendering without browser deps, see leanclient's
    widget_render module which uses quickjs for React SSR.

    Args:
        widget_props: Widget props dict (containing 'html' or 'component')
        width: Viewport width
        height: Viewport height

    Returns:
        Base64-encoded PNG, or None if rendering unavailable/failed
    """
    # Priority: Playwright (most reliable) > html2image (uses system browser)
    try:
        return _render_with_playwright(widget_props, width, height)
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Playwright render failed: {e}")

    try:
        return _render_with_html2image(widget_props, width, height)
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"html2image render failed: {e}")

    logger.debug(
        "No headless browser available. Install: "
        "pip install 'lean-lsp-mcp[render]' && playwright install chromium"
    )
    return None


def _extract_chart_data(widget_props: dict) -> Optional[dict]:
    """Extract chart configuration from widget props."""
    html_prop = widget_props.get("html", {})

    def find_line_chart(elem: Any) -> Optional[dict]:
        if isinstance(elem, dict):
            if "component" in elem:
                comp = elem["component"]
                if isinstance(comp, list) and len(comp) >= 3:
                    name = comp[1]
                    if name == "LineChart":
                        props = comp[2] if len(comp) > 2 else {}
                        children = comp[3] if len(comp) > 3 else []
                        return {"type": "line", "props": props, "children": children}
            if "element" in elem:
                children = elem["element"][2] if len(elem["element"]) > 2 else []
                if isinstance(children, list):
                    for child in children:
                        result = find_line_chart(child)
                        if result:
                            return result
        elif isinstance(elem, list):
            for item in elem:
                result = find_line_chart(item)
                if result:
                    return result
        return None

    return find_line_chart(html_prop)


def _extract_title(widget_props: dict) -> str:
    """Extract title from widget props."""
    html_prop = widget_props.get("html", {})

    def find_title(elem: Any) -> Optional[str]:
        if isinstance(elem, dict):
            if "text" in elem:
                return str(elem["text"])
            if "element" in elem:
                tag = elem["element"][0] if elem["element"] else ""
                if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
                    children = elem["element"][2] if len(elem["element"]) > 2 else []
                    if isinstance(children, list):
                        for child in children:
                            if isinstance(child, dict) and "text" in child:
                                return child["text"]
                children = elem["element"][2] if len(elem["element"]) > 2 else []
                if isinstance(children, list):
                    for child in children:
                        result = find_title(child)
                        if result:
                            return result
        elif isinstance(elem, list):
            for item in elem:
                result = find_title(item)
                if result:
                    return result
        return None

    return find_title(html_prop) or "Chart"


def _build_recharts_html(widget_props: dict, width: int, height: int) -> str:
    """Build HTML page that renders charts using Chart.js."""
    chart_config = _extract_chart_data(widget_props)
    title = _extract_title(widget_props)

    if not chart_config:
        # Fallback: just render title and message
        return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>body {{ background: #1e1e1e; color: #d4d4d4; font-family: sans-serif; padding: 16px; }}</style>
</head><body><h4>{title}</h4><p>Widget contains no chart data</p></body></html>"""

    # Extract data for Chart.js
    data = chart_config["props"].get("data", [])
    x_key = "x"
    y_keys = [k for k in (data[0].keys() if data else []) if k != "x"]

    # Build datasets for Chart.js
    datasets_js = []
    colors = ["#b5de2b", "#8884d8", "#82ca9d", "#ffc658", "#ff7300"]
    for i, y_key in enumerate(y_keys[:5]):
        color = colors[i % len(colors)]
        points = [{"x": d.get(x_key, 0), "y": d.get(y_key, 0)} for d in data]
        datasets_js.append({
            "label": y_key,
            "data": points,
            "borderColor": color,
            "backgroundColor": color,
            "fill": False,
            "tension": 0.1
        })

    chart_data = _serialize_for_js({"datasets": datasets_js})

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
body {{
    margin: 0;
    padding: 16px;
    background: #1e1e1e;
    color: #d4d4d4;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}
h4 {{ margin: 0 0 12px 0; color: #e0e0e0; }}
canvas {{ background: #2d2d2d; border-radius: 4px; }}
</style>
</head>
<body>
<h4>{title}</h4>
<canvas id="chart" width="{width - 32}" height="{height - 80}"></canvas>
<script>
const ctx = document.getElementById('chart').getContext('2d');
new Chart(ctx, {{
    type: 'line',
    data: {chart_data},
    options: {{
        responsive: false,
        plugins: {{
            legend: {{ labels: {{ color: '#d4d4d4' }} }}
        }},
        scales: {{
            x: {{
                type: 'linear',
                title: {{ display: true, text: 'x', color: '#d4d4d4' }},
                ticks: {{ color: '#d4d4d4' }},
                grid: {{ color: '#444' }}
            }},
            y: {{
                title: {{ display: true, text: 'y', color: '#d4d4d4' }},
                ticks: {{ color: '#d4d4d4' }},
                grid: {{ color: '#444' }}
            }}
        }}
    }}
}});
</script>
</body>
</html>"""


def _serialize_for_js(obj: Any) -> str:
    """Serialize Python object to JS-compatible JSON string."""
    import json
    return json.dumps(obj)


def _render_with_playwright(
    widget_props: dict, width: int, height: int
) -> Optional[str]:
    """Render using playwright (chromium)."""
    from playwright.sync_api import sync_playwright

    html_content = _build_recharts_html(widget_props, width, height)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": height})

        # Navigate to allow network requests for CDN scripts
        page.goto("about:blank")
        page.set_content(html_content, wait_until="networkidle")

        # Additional wait for React to render
        page.wait_for_timeout(1000)

        # Wait for recharts wrapper to appear (indicates chart rendered)
        try:
            page.wait_for_selector(".recharts-wrapper", timeout=3000)
        except Exception:
            pass  # Chart may not have rendered, still capture what we have

        screenshot = page.screenshot(type="png")
        browser.close()

        return base64.b64encode(screenshot).decode("utf-8")


def _render_with_html2image(
    widget_props: dict, width: int, height: int
) -> Optional[str]:
    """Render using html2image (uses system Chrome/Edge)."""
    from html2image import Html2Image

    html_content = _build_recharts_html(widget_props, width, height)

    with tempfile.TemporaryDirectory() as tmp_dir:
        hti = Html2Image(output_path=tmp_dir)
        filename = "widget.png"
        hti.screenshot(html_str=html_content, save_as=filename, size=(width, height))

        output_path = os.path.join(tmp_dir, filename)
        if os.path.exists(output_path):
            with open(output_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

    return None
