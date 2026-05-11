"""Registry semantics: registration, lookup, factory, duplicate detection."""

from __future__ import annotations

import pytest

from adaptive_tutor.core.exceptions import ContractError
from adaptive_tutor.core.registry import Registry


def test_register_and_build_roundtrip() -> None:
    reg = Registry()

    @reg.register("widget", "foo")
    class Foo:
        def __init__(self, x: int = 0) -> None:
            self.x = x

    obj = reg.build("widget", {"name": "foo", "x": 7})
    assert isinstance(obj, Foo) and obj.x == 7


def test_register_duplicate_raises() -> None:
    reg = Registry()

    @reg.register("widget", "foo")
    class A: ...

    with pytest.raises(ContractError, match="already bound"):

        @reg.register("widget", "foo")
        class B: ...


def test_get_unknown_raises_with_known_list() -> None:
    reg = Registry()

    @reg.register("widget", "foo")
    class A: ...

    @reg.register("widget", "bar")
    class B: ...

    with pytest.raises(ContractError) as ei:
        reg.get("widget", "baz")
    assert "'foo'" in str(ei.value) and "'bar'" in str(ei.value)


def test_build_missing_name_raises() -> None:
    reg = Registry()
    with pytest.raises(ContractError, match="missing the 'name'"):
        reg.build("widget", {"x": 1})


def test_build_bad_kwargs_raises() -> None:
    reg = Registry()

    @reg.register("widget", "needs_x")
    class A:
        def __init__(self, x: int) -> None:
            self.x = x

    with pytest.raises(ContractError, match="failed to construct"):
        reg.build("widget", {"name": "needs_x"})  # missing x


def test_list_registered_sorted() -> None:
    reg = Registry()

    @reg.register("widget", "zebra")
    class A: ...

    @reg.register("widget", "alpha")
    class B: ...

    assert reg.list_registered("widget") == ["alpha", "zebra"]


def test_categories_isolated() -> None:
    reg = Registry()

    @reg.register("policy", "linear")
    class Pol: ...

    @reg.register("bandit", "linear")
    class Ban: ...

    assert reg.get("policy", "linear") is not reg.get("bandit", "linear")
