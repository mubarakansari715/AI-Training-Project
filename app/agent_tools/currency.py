"""convert_currency(amount, from_currency, to_currency): free rate APIs, no key.

Frankfurter (api.frankfurter.dev) publishes European Central Bank reference
rates, updated once per working day. When it is down or slow (its origin
timed out for most pairs on 2026-11-04), open.er-api.com (ExchangeRate-API's
free endpoint, updated daily) is used instead.
"""

from __future__ import annotations

import logging
import re

import requests

from app.agent_tools.base import Tool, ToolError, ToolInputError, require_number, require_text

logger = logging.getLogger(__name__)

FRANKFURTER_URL = "https://api.frankfurter.dev/v1/latest"
FALLBACK_URL = "https://open.er-api.com/v6/latest/{base}"
# Short, so a slow primary doesn't hold up the agent before the fallback.
TIMEOUT_SECONDS = 5
MAX_AMOUNT = 1e12
_CURRENCY_CODE = re.compile(r"[A-Z]{3}")


def _currency_code(name: str, value) -> str:
    code = require_text(name, value, max_length=10).upper()
    if not _CURRENCY_CODE.fullmatch(code):
        raise ToolInputError(f"{name} must be a 3-letter ISO currency code like USD or INR, got {value!r}")
    return code


def _unsupported(source: str, target: str) -> ToolInputError:
    return ToolInputError(f"Unsupported currency: {source} or {target}")


def _frankfurter_rate(source: str, target: str) -> tuple[float, str]:
    # The rate of 1 unit; requests with an `amount` parameter time out there.
    try:
        response = requests.get(FRANKFURTER_URL, params={"base": source, "symbols": target},
                                timeout=TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise ToolError(f"Frankfurter unreachable: {exc}") from exc
    if response.status_code in (404, 422):
        raise _unsupported(source, target)
    if not response.ok:
        raise ToolError(f"Frankfurter error (HTTP {response.status_code})")
    data = response.json()
    rate = (data.get("rates") or {}).get(target)
    if not isinstance(rate, (int, float)) or rate <= 0:
        raise ToolError(f"Frankfurter has no rate for {source} → {target}")
    return rate, f"ECB reference rate of {data.get('date', 'today')}"


def _fallback_rate(source: str, target: str) -> tuple[float, str]:
    try:
        response = requests.get(FALLBACK_URL.format(base=source), timeout=TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise ToolError(f"open.er-api.com unreachable: {exc}") from exc
    if not response.ok:
        raise ToolError(f"open.er-api.com error (HTTP {response.status_code})")
    data = response.json()
    if data.get("result") != "success":
        if data.get("error-type") == "unsupported-code":
            raise _unsupported(source, target)
        raise ToolError(f"open.er-api.com error: {data.get('error-type', 'unknown')}")
    rate = (data.get("rates") or {}).get(target)
    if rate is None:
        raise _unsupported(source, target)
    if not isinstance(rate, (int, float)) or rate <= 0:
        raise ToolError(f"open.er-api.com has no rate for {source} → {target}")
    updated = " ".join(str(data.get("time_last_update_utc", "")).split()[1:4]) or "today"
    return rate, f"open.er-api.com rate of {updated}"


def convert_currency(amount, from_currency, to_currency) -> str:
    """Convert `amount` of `from_currency` into `to_currency` at today's rate."""
    amount = require_number("amount", amount, MAX_AMOUNT)
    if amount <= 0:
        raise ToolInputError("amount must be greater than 0")
    source = _currency_code("from_currency", from_currency)
    target = _currency_code("to_currency", to_currency)
    if source == target:
        return f"{amount:,.2f} {source} = {amount:,.2f} {target} (same currency)"

    try:
        rate, origin = _frankfurter_rate(source, target)
    except ToolError as primary:
        logger.info("Frankfurter failed (%s); trying open.er-api.com", primary)
        try:
            rate, origin = _fallback_rate(source, target)
        except ToolError as fallback:
            raise ToolError(f"Currency services unavailable ({primary}; {fallback}). "
                            "Don't retry; tell the user.") from fallback
    return f"{amount:,.2f} {source} = {amount * rate:,.2f} {target} (rate {rate:.6g}, {origin})"


CURRENCY_TOOL = Tool(
    name="convert_currency",
    description="Convert an amount of money from one currency to another using today's exchange rate.",
    parameters={
        "type": "object",
        "properties": {
            "amount": {"type": "number", "description": "The amount to convert, greater than 0."},
            "from_currency": {"type": "string", "description": "3-letter ISO code to convert from, e.g. USD."},
            "to_currency": {"type": "string", "description": "3-letter ISO code to convert to, e.g. INR."},
        },
        "required": ["amount", "from_currency", "to_currency"],
    },
    function=convert_currency,
)
