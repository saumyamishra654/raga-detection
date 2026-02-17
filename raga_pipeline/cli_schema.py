"""
Utilities for converting argparse configuration into UI-ready schemas.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Set

from .config import build_cli_parser


MODES: List[str] = ["preprocess", "detect", "analyze"]

ADVANCED_DESTS: Set[str] = {
    "separator",
    "demucs_model",
    "source_type",
    "melody_source",
    "vocalist_gender",
    "instrument_type",
    "fmin_note",
    "fmax_note",
    "vocal_confidence",
    "accomp_confidence",
    "prominence_high",
    "prominence_low",
    "bias_rotation",
    "force",
    "skip_separation",
    "raga_db",
}


def _infer_action_type(action: argparse.Action) -> str:
    if isinstance(action, argparse._StoreTrueAction):
        return "store_true"
    if isinstance(action, argparse._StoreFalseAction):
        return "store_false"
    return "store"


def _infer_value_type(action: argparse.Action) -> str:
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        return "bool"
    if action.type is int:
        return "int"
    if action.type is float:
        return "float"
    return "str"


def _first_long_flag(option_strings: List[str]) -> Optional[str]:
    for flag in option_strings:
        if flag.startswith("--"):
            return flag
    return option_strings[0] if option_strings else None


def _get_subparser_action(parser: argparse.ArgumentParser) -> argparse._SubParsersAction:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    raise ValueError("No subparsers found in parser.")


def _dest_set_for_mode(parser: argparse.ArgumentParser, mode: str) -> Set[str]:
    subparsers = _get_subparser_action(parser)
    if mode not in subparsers.choices:
        raise ValueError(f"Unknown mode: {mode}")
    mode_parser = subparsers.choices[mode]
    return {
        action.dest
        for action in mode_parser._actions
        if action.dest not in {"help", "command"}
        and not isinstance(action, argparse._SubParsersAction)
    }


def _common_dest_set(parser: argparse.ArgumentParser) -> Set[str]:
    detect_dests = _dest_set_for_mode(parser, "detect")
    analyze_dests = _dest_set_for_mode(parser, "analyze")
    # tonic/raga have different semantics across detect/analyze, so keep them mode-specific.
    return (detect_dests & analyze_dests) - {"tonic", "raga"}


def list_modes() -> List[str]:
    """Return supported pipeline command modes."""
    return list(MODES)


def get_mode_schema(mode: str) -> Dict[str, Any]:
    """
    Build UI schema metadata for a pipeline mode from argparse definitions.

    Returns a dict with fields including:
      - name, flags, flag, mode, required, default, choices, help, action, value_type, group
    """
    if mode not in MODES:
        raise ValueError(f"Unsupported mode '{mode}'. Expected one of: {', '.join(MODES)}")

    parser = build_cli_parser()
    subparsers = _get_subparser_action(parser)
    mode_parser = subparsers.choices.get(mode)
    if mode_parser is None:
        raise ValueError(f"Parser does not define mode '{mode}'.")

    common_dests = _common_dest_set(parser)
    mode_dests = _dest_set_for_mode(parser, mode)

    fields: List[Dict[str, Any]] = []
    required: List[str] = []

    for action in mode_parser._actions:
        if action.dest in {"help", "command"}:
            continue
        if isinstance(action, argparse._SubParsersAction):
            continue

        option_strings = list(action.option_strings)
        primary_flag = _first_long_flag(option_strings)
        if primary_flag is None:
            continue

        action_type = _infer_action_type(action)
        value_type = _infer_value_type(action)
        default_val: Any = None if action.default is argparse.SUPPRESS else action.default
        choices = list(action.choices) if action.choices is not None else None

        if action.dest in ADVANCED_DESTS:
            group = "advanced"
        elif action.dest in common_dests and action.dest in mode_dests:
            group = "common"
        else:
            group = "mode"

        field_obj: Dict[str, Any] = {
            "name": action.dest,
            "flags": option_strings,
            "flag": primary_flag,
            "mode": mode,
            "required": bool(action.required),
            "default": default_val,
            "choices": choices,
            "help": action.help or "",
            "action": action_type,
            "value_type": value_type,
            "group": group,
        }
        fields.append(field_obj)

        if action.required:
            required.append(action.dest)

    return {
        "mode": mode,
        "fields": fields,
        "required": required,
        "groups": ["common", "mode", "advanced"],
    }

