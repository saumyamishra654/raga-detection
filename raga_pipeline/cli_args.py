"""
Helpers to convert structured UI payloads into pipeline argv lists.
"""

from __future__ import annotations

import shlex
from typing import Any, Dict, Iterable, List, Sequence

from .cli_schema import get_mode_schema, list_modes


def _coerce_extra_args(extra_args: Any) -> List[str]:
    if extra_args is None:
        return []
    if isinstance(extra_args, str):
        return shlex.split(extra_args)
    if isinstance(extra_args, Sequence):
        return [str(x) for x in extra_args if str(x).strip()]
    raise TypeError("extra_args must be a string, sequence, or None.")


def _normalize_scalar(value: Any) -> str:
    return str(value)


def params_to_argv(mode: str, params: Dict[str, Any], extra_args: Any = None) -> List[str]:
    """
    Convert UI form parameters to CLI-style argv.

    Args:
        mode: preprocess | detect | analyze
        params: dict keyed by argparse destination names
        extra_args: optional raw args (string or list) appended as-is
    """
    if mode not in list_modes():
        raise ValueError(f"Unsupported mode '{mode}'.")
    if not isinstance(params, dict):
        raise TypeError("params must be a dictionary.")

    schema = get_mode_schema(mode)
    argv: List[str] = [mode]

    for field in schema["fields"]:
        name = field["name"]
        if name not in params:
            continue

        value = params[name]
        action = field["action"]
        flag = field["flag"]

        if flag is None:
            continue

        if action == "store_true":
            if bool(value):
                argv.append(flag)
            continue

        if action == "store_false":
            # For flags like --bias-rotation where the flag flips a default True -> False.
            if value is False:
                argv.append(flag)
            continue

        # "store" action
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue

        argv.extend([flag, _normalize_scalar(value)])

    argv.extend(_coerce_extra_args(extra_args))
    return argv

