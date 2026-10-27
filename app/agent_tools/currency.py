"""convert_currency(amount, from_currency, to_currency) via the Frankfurter API.

Frankfurter (api.frankfurter.dev) publishes European Central Bank reference
rates: free, no API key, updated once per working day.
"""

from __future__ import annotations

import re

import requests

from app.agent_tools.base import HTTP_TIMEOUT_SECONDS, Tool, ToolError, ToolInputError, require_number, require_text

FRANKFURTER_URL = "https://api.frankfurter.dev/v1/latest"
MAX_AMOUNT = 1e12
_CURRENCY_CODE = re.compile(r"[A-Z]{3}")


def _currency_code(name: str, value) -> str:
    code = require_text(name, value, max_length=10).upper()
    if not _CURRENCY_CODE.fullmatch(code):
        raise ToolInputError(f"{name} must be a 3-letter ISO currency code like USD or INR, got {value!r}")
    return code


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
        response = requests.get(
            FRANKFURTER_URL, params={"base": source, "symbols": target, "amount": amount},
            timeout=HTTP_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise ToolError(f"Currency service unreachable: {exc}") from exc
    if response.status_code in (404, 422):
        raise ToolInputError(f"Unsupported currency: {source} or {target}")
    if not response.ok:
        raise ToolError(f"Currency service error (HTTP {response.status_code})")

    data = response.json()
    converted = (data.get("rates") or {}).get(target)
    if converted is None:
        raise ToolError(f"No rate available for {source} → {target}")
    return (f"{amount:,.2f} {source} = {converted:,.2f} {target} "
            f"(rate {converted / amount:.6g}, ECB reference rate of {data.get('date', 'today')})")


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
