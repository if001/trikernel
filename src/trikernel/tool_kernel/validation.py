from __future__ import annotations

from typing import Any, Dict


def validate_input(schema: Dict[str, Any], args: Dict[str, Any]) -> None:
    _validate(schema, args, path="$")


def _validate(schema: Dict[str, Any], value: Any, *, path: str) -> None:
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path} must be {schema['const']}")

    expected_type = schema.get("type")
    if expected_type:
        _validate_type(expected_type, value, path=path)

    if expected_type == "object":
        _validate_object(schema, value, path=path)
    elif expected_type == "array":
        _validate_array(schema, value, path=path)


def _validate_type(expected_type: str, value: Any, *, path: str) -> None:
    if expected_type == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be an object")
    elif expected_type == "string":
        if not isinstance(value, str):
            raise ValueError(f"{path} must be a string")
    elif expected_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{path} must be an integer")
    elif expected_type == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{path} must be a number")
    elif expected_type == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"{path} must be a boolean")
    elif expected_type == "array":
        if not isinstance(value, list):
            raise ValueError(f"{path} must be an array")


def _validate_object(schema: Dict[str, Any], value: Any, *, path: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    required = schema.get("required", [])
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(f"{path} missing required keys: {missing}")
    properties = schema.get("properties", {})
    for key, prop_schema in properties.items():
        if key in value:
            _validate(prop_schema, value[key], path=f"{path}.{key}")


def _validate_array(schema: Dict[str, Any], value: Any, *, path: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array")
    items = schema.get("items")
    if not items:
        return
    for idx, item in enumerate(value):
        _validate(items, item, path=f"{path}[{idx}]")
