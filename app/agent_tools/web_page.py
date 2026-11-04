"""read_webpage(url): the readable text of a web page, so the agent can check a
fact on the page itself instead of guessing from a search snippet.

Guarded because the URL comes from the model (and may come from text it read):
only http(s), only public addresses (no localhost or private network), at most
`MAX_BYTES` downloaded, and only HTML or plain text.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from urllib.parse import urlparse

import lxml.html
import requests

from app.agent_tools.base import HTTP_TIMEOUT_SECONDS, Tool, ToolError, ToolInputError, require_text

MAX_URL_LENGTH = 2000
MAX_BYTES = 2_000_000
# Characters of page text returned to the model (about 1,500 tokens).
MAX_CHARS = 6000
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/130.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
    "Accept-Language": "en;q=0.9",
}
_BLOCK_TAGS = ("p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article",
               "table", "ul", "ol", "dt", "dd", "blockquote", "pre")
_CELL_TAGS = ("td", "th")
_SKIP_TAGS = ("script", "style", "noscript", "svg", "nav", "footer", "header", "aside", "form", "iframe")


def _check_url(url: str) -> str:
    url = require_text("url", url, MAX_URL_LENGTH)
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ToolInputError(f"url must be a full http(s) address, got {url!r}")
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(parsed.hostname, None)}
    except socket.gaierror as exc:
        raise ToolError(f"Could not find the site {parsed.hostname}") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address.split("%")[0])
        if not ip.is_global:
            raise ToolInputError("url must point to a public website")
    return url


def page_text(html: str) -> tuple[str, str]:
    """(title, readable text) of an HTML page."""
    tree = lxml.html.fromstring(html)
    title = " ".join((tree.findtext(".//title") or "").split())
    for element in tree.xpath("|".join(f"//{tag}" for tag in _SKIP_TAGS)):
        element.drop_tree()
    root = (tree.xpath("//main") or tree.xpath("//article") or tree.xpath("//body") or [tree])[0]
    # text_content() joins everything; keep paragraphs, headings and table cells apart.
    for element in root.iter(*_BLOCK_TAGS, *_CELL_TAGS):
        element.tail = ("\n" if element.tag in _BLOCK_TAGS else " | ") + (element.tail or "")
    lines = (" ".join(line.split()) for line in root.text_content().splitlines())
    return title, "\n".join(line for line in lines if len(line) > 1)


def read_webpage(url) -> str:
    """The title and main text of `url` (cut to MAX_CHARS)."""
    url = _check_url(url)
    try:
        with requests.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT_SECONDS, stream=True,
                          allow_redirects=True) as response:
            if not response.ok:
                raise ToolError(f"The page returned HTTP {response.status_code}")
            kind = response.headers.get("Content-Type", "").lower()
            if kind and not any(t in kind for t in ("html", "text/plain", "xml")):
                raise ToolError(f"Not a web page ({kind.split(';')[0]})")
            body = b""
            for chunk in response.iter_content(65536):
                body += chunk
                if len(body) > MAX_BYTES:
                    break
            encoding = response.encoding or "utf-8"
    except requests.RequestException as exc:
        raise ToolError(f"Could not open the page: {exc}") from exc

    text = body.decode(encoding, errors="replace")
    title = ""
    if "html" in kind or re.search(r"<html|<body", text[:2000], re.IGNORECASE):
        try:
            title, text = page_text(text)
        except (ValueError, lxml.etree.ParserError) as exc:
            raise ToolError("Could not read the page's text") from exc
    text = text.strip()
    if not text:
        raise ToolError("The page has no readable text (it may need JavaScript)")
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + " …[cut]"
    return f"{title or url}\n{url}\n\n{text}"


WEB_PAGE_TOOL = Tool(
    name="read_webpage",
    description=(
        "Open a web page (usually a link from web_search) and return its main text, to check "
        "a fact on the page itself when the search snippet doesn't clearly state it."
    ),
    parameters={
        "type": "object",
        "properties": {"url": {"type": "string", "description": "Full http(s) address of the page."}},
        "required": ["url"],
    },
    function=read_webpage,
)
