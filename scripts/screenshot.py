"""Capture a screenshot of the Streamlit UI for docs/.

Usage:
    # 1. Start the UI on a known port
    uv run streamlit run src/desktop_search/app.py \\
        --server.headless true --server.port 8765 \\
        --browser.gatherUsageStats false &

    # 2. Wait until it's serving, then snap
    uv run python scripts/screenshot.py http://localhost:8765 docs/ui-empty.png

Requires the `playwright` dev dep and `playwright install chromium`.
"""

import sys

from playwright.sync_api import sync_playwright


def capture(url: str, out_path: str) -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        page = ctx.new_page()
        page.goto(url, wait_until="networkidle", timeout=60_000)
        page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=30_000)
        page.wait_for_timeout(1500)
        page.screenshot(path=out_path, full_page=False)
        browser.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/screenshot.py <url> <out.png>", file=sys.stderr)
        sys.exit(2)
    capture(sys.argv[1], sys.argv[2])
