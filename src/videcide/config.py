"""Config helpers: YAML/JSON configs with ``_base_`` inheritance, and YAML defaults for argparse scripts."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence


def load_config(path: str | Path):
    """Load a YAML or JSON config as an OmegaConf DictConfig.

    A top-level ``_base_`` entry (a path relative to the file, or a list of such paths) is loaded first, recursively,
    and the file's own values are merged on top of it.
    """
    from omegaconf import OmegaConf

    path = Path(path).resolve()
    config = OmegaConf.load(path)
    bases = config.pop("_base_", None)
    if bases is None:
        return config
    merged = OmegaConf.create()
    for base in [bases] if isinstance(bases, str) else list(bases):
        merged = OmegaConf.merge(merged, load_config(path.parent / str(base)))
    return OmegaConf.merge(merged, config)


def load_config_dict(path: str | Path) -> dict[str, Any]:
    """Like :func:`load_config`, resolved into plain Python containers."""
    from omegaconf import OmegaConf

    return OmegaConf.to_container(load_config(path), resolve=True)


def parse_args_with_config(parser: argparse.ArgumentParser, argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse ``argv`` where the keys of an optional ``--config`` YAML file act as defaults of the options.

    Keys are option names (``sample_steps`` or ``sample-steps``); explicit command-line options take precedence.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path)
    known, _ = pre.parse_known_args(argv)
    parser.add_argument("--config", type=Path, help="YAML file whose keys set the defaults of the options below")
    if known.config is None:
        return parser.parse_args(argv)
    import yaml

    values = {str(k).replace("-", "_"): v for k, v in (yaml.safe_load(known.config.read_text()) or {}).items()}
    actions = {action.dest: action for action in parser._actions}
    unknown = sorted(set(values) - set(actions))
    if unknown:
        parser.error(f"unknown keys in {known.config}: {unknown}")
    for dest in values:
        actions[dest].required = False
    parser.set_defaults(**values)
    args = parser.parse_args(argv)
    for dest in values:
        choices = actions[dest].choices
        if choices is not None and getattr(args, dest) not in choices:
            parser.error(f"{known.config}: {dest}={getattr(args, dest)!r} is not one of {list(choices)}")
    for dest in values:  # argparse converts string defaults only; also convert lists and non-string scalars
        action, value = actions[dest], getattr(args, dest)
        convert = action.type
        if convert is None or value is None:
            continue
        if isinstance(value, list):
            setattr(args, dest, [convert(v) if isinstance(v, (str, int, float)) else v for v in value])
        elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
            setattr(args, dest, convert(value))
    return args


def missing_values(config, prefix: str = "") -> list[str]:
    """Keys whose value is the mandatory placeholder ``???`` (interpolations are not resolved)."""
    from omegaconf import DictConfig, OmegaConf

    missing = []
    for key in config.keys():
        if OmegaConf.is_missing(config, key):
            missing.append(f"{prefix}{key}")
        elif not OmegaConf.is_interpolation(config, key) and isinstance(config[key], DictConfig):
            missing += missing_values(config[key], f"{prefix}{key}.")
    return missing
