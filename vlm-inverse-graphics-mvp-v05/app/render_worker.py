"""Isolated SVG renderer so Cairo/Chromium state never leaks across jobs.

Run: python -m app.render_worker input.svg output.png
"""
from __future__ import annotations
import sys
from pathlib import Path
import os


def render(svg: Path, png: Path) -> str:
    try:
        from playwright.sync_api import sync_playwright
        # Do not start the Playwright driver when Chromium is not installed.
        # Failed launches can leave child processes around on some Linux setups.
        with sync_playwright() as check:
            if not Path(check.chromium.executable_path).is_file():
                raise FileNotFoundError('Chromium not installed')
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
            try:
                page=browser.new_page(viewport={'width':1000,'height':1000},device_scale_factor=1)
                page.goto(svg.resolve().as_uri(),wait_until='load',timeout=30000)
                page.screenshot(path=str(png),full_page=False)
            finally:
                browser.close()
        return 'playwright'
    except Exception as browser_exc:
        try:
            import cairosvg
            cairosvg.svg2png(url=str(svg),write_to=str(png),output_width=1000,output_height=1000)
            return 'cairosvg_fallback'
        except Exception as fallback_exc:
            raise RuntimeError('SVG rasterization unavailable. Run `python -m playwright install chromium` '
                               f'or install CairoSVG. Browser: {type(browser_exc).__name__}, '
                               f'CairoSVG: {type(fallback_exc).__name__}') from fallback_exc


if __name__=='__main__':
    if len(sys.argv) not in (3,4):
        raise SystemExit('Usage: python -m app.render_worker input.svg output.png')
    result=render(Path(sys.argv[1]),Path(sys.argv[2]))
    if len(sys.argv)==4:
        Path(sys.argv[3]).write_text(result,encoding='utf-8')
    else:
        print(result)
