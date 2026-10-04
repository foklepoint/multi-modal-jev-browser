"""Draw numbered boxes over the page so one screenshot can show what the element list is talking about.

The overlay goes into the DOM, the screenshot is taken with it in place, and the overlay is removed
again, so the page a person would see is never left changed.
"""
from __future__ import annotations

from typing import Any

OVERLAY_ID = "__mmjb_marks"

PALETTE = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#008080",
    "#9a6324",
    "#800000",
]

RECTS_JS = r"""
((indices) => {
  const nodes = (window.__mmjb && window.__mmjb.nodes) || {};
  const out = [];
  for (const index of indices) {
    const element = nodes[index];
    if (!element || !element.isConnected) continue;
    const rect = element.getBoundingClientRect();
    if (rect.width < 1 || rect.height < 1) continue;
    out.push([index, rect.left, rect.top, rect.width, rect.height]);
  }
  return out;
})
"""

DRAW_JS = r"""
((items) => new Promise((resolve) => {
  const old = document.getElementById('__mmjb_marks');
  if (old) old.remove();
  const layer = document.createElement('div');
  layer.id = '__mmjb_marks';
  layer.style.cssText = 'position:fixed;left:0;top:0;width:100vw;height:100vh;pointer-events:none;z-index:2147483647';
  const colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#008080', '#9a6324', '#800000'];
  for (const item of items) {
    const index = item[0];
    const x = item[1];
    const y = item[2];
    const width = item[3];
    const height = item[4];
    const color = colors[Number(index) % colors.length];
    const frame = document.createElement('div');
    frame.style.cssText = 'position:fixed;left:' + x + 'px;top:' + y + 'px;width:' + width + 'px;height:' + height +
      'px;border:2px solid ' + color + ';box-sizing:border-box;pointer-events:none';
    const tag = document.createElement('div');
    tag.textContent = String(index);
    tag.style.cssText = 'position:absolute;left:-2px;top:' + (y < 18 ? '0px' : '-16px') +
      ';min-width:16px;height:16px;padding:0 3px;font:bold 11px/16px Arial,sans-serif;color:#fff;background:' + color +
      ';text-align:center;box-sizing:border-box;border-radius:2px';
    frame.appendChild(tag);
    layer.appendChild(frame);
  }
  document.documentElement.appendChild(layer);
  window.scrollTo(window.scrollX, window.scrollY);
  requestAnimationFrame(() => requestAnimationFrame(() => resolve(true)));
}))
"""

REMOVE_JS = r"""
(() => {
  const layer = document.getElementById('__mmjb_marks');
  if (layer) layer.remove();
  return true;
})()
"""


async def draw_marks(page: Any, indices: list[int]) -> list[list[Any]]:
    """Put the numbered boxes on the page and return the rectangles they were drawn at."""
    if not indices:
        return []
    rects = await page.evaluate(RECTS_JS, list(indices))
    if not rects:
        return []
    await page.evaluate(DRAW_JS, rects)
    return rects


async def remove_marks(page: Any) -> None:
    """Take the boxes off the page. Safe to call when there were none."""
    try:
        await page.evaluate(REMOVE_JS)
    except Exception:
        pass


async def marked_screenshot(page: Any, indices: list[int], quality: int = 72) -> bytes | None:
    """A JPEG of the viewport with one coloured box and index tag per element, or None if it cannot be drawn."""
    try:
        rects = await draw_marks(page, indices)
        if not rects:
            return None
        raw = await page.screenshot(type="jpeg", quality=quality, animations="disabled", caret="hide")
        return raw
    except Exception:
        return None
    finally:
        await remove_marks(page)