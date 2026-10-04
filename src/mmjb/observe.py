"""Read the page: the interactive elements in the viewport, the page text and the page's own facts.

The JavaScript in this module runs in the page and returns one plain dict. The Python side turns it
into typed objects so the rest of the library never touches a dict it did not build.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any

MAX_TEXT = 12000
MAX_ELEMENTS = 300
MAX_OFFSCREEN = 25
MAX_OPTIONS = 60

OBSERVE_JS = r"""
(() => {
  const clean = (value) => (value || '').replace(/\s+/g, ' ').trim();
  const SELECTOR = [
    'a[href]', 'button', 'input', 'textarea', 'select', 'summary',
    '[contenteditable=""]', '[contenteditable="true"]', '[contenteditable="plaintext-only"]',
    '[role=button]', '[role=link]', '[role=checkbox]', '[role=radio]', '[role=switch]',
    '[role=tab]', '[role=menuitem]', '[role=option]', '[role=combobox]', '[role=textbox]'
  ].join(', ');
  const TYPEABLE_ROLES = ['textbox', 'searchbox', 'password', 'combobox', 'spinbutton', 'slider'];

  const visible = (element) => {
    if (element.tagName === 'INPUT' && (element.type || '').toLowerCase() === 'hidden') return false;
    const style = getComputedStyle(element);
    if (style.visibility === 'hidden' || style.display === 'none') return false;
    if (parseFloat(style.opacity) === 0) return false;
    if (element.checkVisibility) {
      try {
        if (!element.checkVisibility({ checkVisibilityCSS: true, checkOpacity: true })) return false;
      } catch (error) {
        if (!element.checkVisibility()) return false;
      }
    }
    const rect = element.getBoundingClientRect();
    if (rect.width < 1 || rect.height < 1) return false;
    return true;
  };
  const rect = (element) => element.getBoundingClientRect();
  const onScreen = (element) => {
    const box = rect(element);
    return box.bottom > 0 && box.top < window.innerHeight && box.right > 0 && box.left < window.innerWidth;
  };

  const labelText = (element) => {
    let text = clean(element.getAttribute('aria-label'));
    if (text) return text;
    const labelledby = clean(element.getAttribute('aria-labelledby'));
    if (labelledby) {
      const parts = [];
      for (const id of labelledby.split(/\s+/)) {
        const node = document.getElementById(id);
        if (node) parts.push(clean(node.innerText || node.textContent));
      }
      text = clean(parts.join(' '));
      if (text) return text;
    }
    if (element.labels && element.labels.length) {
      text = clean(Array.from(element.labels).map((label) => clean(label.innerText || label.textContent)).join(' '));
      if (text) return text;
    }
    text = clean(element.innerText);
    if (text) return text;
    text = clean(element.getAttribute('placeholder')) || clean(element.getAttribute('data-placeholder'));
    if (text) return text;
    text = clean(element.getAttribute('title'));
    if (text) return text;
    text = clean(element.getAttribute('name')) || clean(element.getAttribute('id'));
    if (text) return text;
    if (element.tagName === 'INPUT') return clean(element.value) || clean(element.type);
    return clean(element.tagName.toLowerCase());
  };

  const roleOf = (element) => {
    const explicit = clean(element.getAttribute('role'));
    if (explicit) return explicit;
    const tag = element.tagName.toLowerCase();
    if (tag === 'a') return element.hasAttribute('href') ? 'link' : '';
    if (tag === 'button' || tag === 'summary') return 'button';
    if (tag === 'textarea') return 'textbox';
    if (tag === 'select') return element.multiple || element.size > 1 ? 'listbox' : 'combobox';
    if (tag === 'input') {
      const type = (element.type || 'text').toLowerCase();
      if (type === 'checkbox') return 'checkbox';
      if (type === 'radio') return 'radio';
      if (type === 'file') return 'file-input';
      if (type === 'submit' || type === 'button' || type === 'reset' || type === 'image') return 'button';
      if (type === 'password') return 'password';
      if (type === 'search') return 'searchbox';
      if (type === 'range') return 'slider';
      if (type === 'number') return 'spinbutton';
      return 'textbox';
    }
    return '';
  };

  const editable = (element) => {
    if (element.isContentEditable) return element.closest('[contenteditable]') === element;
    return false;
  };

  const kindOf = (element, role) => {
    const tag = element.tagName.toLowerCase();
    if (tag === 'select') return 'select';
    if (tag === 'input' && (element.type || '').toLowerCase() === 'file') return 'upload';
    if (editable(element) || tag === 'textarea') return 'type';
    if (tag === 'input' && TYPEABLE_ROLES.indexOf(role) !== -1) return 'type';
    if (TYPEABLE_ROLES.indexOf(role) !== -1) return 'type';
    return 'click';
  };

  const valueOf = (element, kind) => {
    if (kind === 'select') {
      const chosen = Array.from(element.selectedOptions || []);
      if (chosen.length === 0) return '';
      // A placeholder option (value="") selected by default is not a choice, so it reads as empty.
      if (chosen.length === 1 && chosen[0].value === '') return '';
      return clean(chosen.map((option) => option.textContent).join(', '));
    }
    const raw = element.value === undefined ? clean(element.innerText) : String(element.value);
    if ((element.type || '').toLowerCase() === 'password') return raw ? '********' : '';
    return clean(raw).slice(0, 300);
  };

  const isDisabled = (element) => {
    if (element.disabled === true) return true;
    if (element.getAttribute('aria-disabled') === 'true') return true;
    return !!element.closest('[disabled], [aria-disabled="true"]');
  };

  const stateOf = (element) => {
    if ((element.type || '').toLowerCase() === 'checkbox') return !!element.checked;
    if ((element.type || '').toLowerCase() === 'radio') return !!element.checked;
    const aria = element.getAttribute('aria-checked');
    if (aria === 'true') return true;
    if (aria === 'false') return false;
    return null;
  };

  const optionTexts = (element) => {
    if (element.tagName !== 'SELECT') return [];
    const out = [];
    for (const option of Array.from(element.options)) {
      const text = clean(option.textContent);
      if (text) out.push(text.slice(0, 80));
      if (out.length >= 60) break;
    }
    return out;
  };

  const candidates = Array.from(document.querySelectorAll(SELECTOR));
  const store = {};
  const elements = [];
  const offscreen = [];
  const disabledControls = [];
  const controlState = [];
  let index = 0;
  let dropped = 0;

  for (const element of candidates) {
    if (!visible(element)) continue;
    const onscreen = onScreen(element);
    const label = labelText(element).slice(0, 160);
    const role = roleOf(element);
    const kind = kindOf(element, role);
    const disabled = isDisabled(element);
    if (!onscreen) {
      if (label && offscreen.length < 25) offscreen.push({ index: offscreen.length + 1, label: label, kind: kind, direction: rect(element).top < 0 ? 'above' : 'below' });
      continue;
    }
    if (disabled) {
      if (label && disabledControls.indexOf(label) === -1 && disabledControls.length < 12) disabledControls.push(label);
    }
    if (kind === 'select' || kind === 'type' || kind === 'upload' || role === 'checkbox' || role === 'radio' || role === 'switch') {
      controlState.push([label, valueOf(element, kind), stateOf(element)]);
    }
    index = index + 1;
    if (index > 300) {
      dropped = dropped + 1;
      continue;
    }
    store[index] = element;
    const record = {
      index: index,
      kind: kind,
      role: role || kind,
      label: label,
      value: valueOf(element, kind),
      disabled: disabled,
      required: element.required === true || element.getAttribute('aria-required') === 'true'
    };
    const checked = stateOf(element);
    if (checked === true || checked === false) record.checked = checked;
    if (kind === 'select') record.options = optionTexts(element);
    elements.push(record);
  }

  window.__mmjb = { nodes: store, seen: index };
  const bodyText = (document.body ? document.body.innerText : '') || '';
  return {
    url: location.href,
    title: (document.title || '').slice(0, 300),
    elements: elements,
    offscreen_controls: offscreen,
    disabled_controls: disabledControls,
    control_state: controlState,
    text: bodyText.slice(0, 12000),
    text_length: bodyText.length,
    scroll: {
      x: Math.round(window.scrollX),
      y: Math.round(window.scrollY),
      page_height: Math.round(document.documentElement.scrollHeight),
      viewport_width: Math.round(window.innerWidth),
      viewport_height: Math.round(window.innerHeight)
    },
    element_count: index,
    dropped_elements: dropped
  };
})()
"""


@dataclass
class Element:
    """One interactive element the page shows right now, numbered from 1."""

    index: int
    kind: str
    role: str
    label: str
    value: str = ""
    disabled: bool = False
    required: bool = False
    checked: bool | None = None
    options: list[str] = field(default_factory=list)
    clicked_without_effect: int = 0
    options_removed: list[str] = field(default_factory=list)

    def as_state(self) -> dict[str, Any]:
        """The element as the decision models see it."""
        record: dict[str, Any] = {
            "index": self.index,
            "kind": self.kind,
            "role": self.role,
            "label": self.label,
            "value": self.value,
            "disabled": self.disabled,
            "required": self.required,
        }
        if self.checked is not None:
            record["checked"] = self.checked
        if self.options:
            record["options"] = self.options
        if self.clicked_without_effect:
            record["clicked_without_effect"] = self.clicked_without_effect
        return record

    def describe(self) -> str:
        """One line for a human or an agent reading a terminal."""
        bits = [f"[{self.index}] {self.kind} {self.role} {self.label}"]
        if self.value:
            bits.append(f"= {self.value[:40]}")
        if self.checked is not None:
            bits.append(f"(checked={self.checked})")
        if self.disabled:
            bits.append("(disabled)")
        if self.required:
            bits.append("(required)")
        if self.clicked_without_effect:
            bits.append(f"(clicked {self.clicked_without_effect}x with no effect)")
        return " ".join(bits)


@dataclass
class Observation:
    """What the page looks like at one moment."""

    url: str
    title: str
    text: str
    elements: list[Element]
    offscreen_controls: list[dict[str, Any]]
    disabled_controls: list[str]
    control_state: list[list[Any]]
    scroll: dict[str, Any]
    text_length: int = 0
    element_count: int = 0
    dropped_elements: int = 0

    def by_index(self, index: int) -> Element | None:
        for element in self.elements:
            if element.index == index:
                return element
        return None

    def as_state(self) -> dict[str, Any]:
        """The page as the decision models see it, with nothing browser specific in it."""
        return {
            "url": self.url,
            "title": self.title,
            "text": self.text[:MAX_TEXT],
            "elements": [element.as_state() for element in self.elements],
            "offscreen_controls": self.offscreen_controls,
            "disabled_controls": self.disabled_controls,
            "scroll": self.scroll,
        }

    def fingerprint(self) -> str:
        """A cheap identity for the page, used to notice that a click did nothing."""
        parts = [self.url, self.title, str(self.text_length)]
        for row in self.control_state:
            parts.append(str(row))
        return "|".join(parts)

    def listing(self) -> str:
        """The numbered element list as text, for the CLI and the MCP look tool."""
        lines = [f"URL: {self.url}", f"Title: {self.title}", ""]
        for element in self.elements:
            lines.append(element.describe())
        if self.offscreen_controls:
            names = ", ".join(f"{item['index']} {item['label']}" for item in self.offscreen_controls)
            lines.append("")
            lines.append("Off screen: " + names)
        if self.disabled_controls:
            lines.append("Disabled until the form is complete: " + "; ".join(self.disabled_controls))
        return "\n".join(lines)


def _row_to_element(record: dict[str, Any]) -> Element:
    element = Element(
        index=int(record.get("index") or 0),
        kind=str(record.get("kind") or "click"),
        role=str(record.get("role") or ""),
        label=str(record.get("label") or ""),
        value=str(record.get("value") or ""),
        disabled=bool(record.get("disabled")),
        required=bool(record.get("required")),
    )
    if "checked" in record:
        element.checked = bool(record.get("checked"))
    if record.get("options"):
        element.options = [str(option) for option in record.get("options")]
    return element


def observation_from_dict(raw: dict[str, Any]) -> Observation:
    """Turn the page's own JSON into typed objects."""
    elements = []
    for record in raw.get("elements") or []:
        elements.append(_row_to_element(record))
    return Observation(
        url=str(raw.get("url") or ""),
        title=str(raw.get("title") or ""),
        text=str(raw.get("text") or "")[:MAX_TEXT],
        elements=elements,
        offscreen_controls=list(raw.get("offscreen_controls") or []),
        disabled_controls=[str(item) for item in (raw.get("disabled_controls") or [])],
        control_state=[list(row) for row in (raw.get("control_state") or [])],
        scroll=dict(raw.get("scroll") or {}),
        text_length=int(raw.get("text_length") or 0),
        element_count=int(raw.get("element_count") or len(elements)),
        dropped_elements=int(raw.get("dropped_elements") or 0),
    )


async def observe(page: Any) -> Observation:
    """Read the page that is in front of the browser right now."""
    raw = await page.evaluate(OBSERVE_JS)
    if not isinstance(raw, dict):
        raise RuntimeError("the page returned no observation; the tab may have navigated")
    return observation_from_dict(raw)


def data_uri(raw_bytes: bytes, kind: str = "jpeg") -> str:
    """A base64 data URI, the form every provider here wants images in."""
    encoded = base64.b64encode(raw_bytes).decode("ascii")
    return f"data:image/{kind};base64,{encoded}"