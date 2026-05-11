"""Decorator-based plug-in registry.

What & why
----------
Every later phase ships new components — replay buffers, reward
components, curriculum providers, policies, critics, masks. We do *not*
want a giant ``if/elif`` switch in the orchestration code that maps
config strings to classes. Instead, each component registers itself at
import time::

    from adaptive_tutor.core.registry import register

    @register("replay", name="uniform")
    class UniformReplay: ...

    @register("replay", name="per")
    class PrioritizedReplay: ...

Then the orchestrator instantiates by name::

    replay = build_from_config("replay", {"name": "per",
                                          "capacity": 100_000})

The registry is *typed* per category (``"replay"``, ``"policy"``,
``"curriculum"``, ``"reward"``, ...). Each category has its own
namespace so the same short name (``"linear"``) can refer to a linear
policy and a linear bandit without collision.

Determinism note
----------------
Registration order is preserved (Python ``dict`` is insertion-ordered).
We export ``list_registered(category)`` so tests can pin the set of
registered components — surprise additions to a category from a
third-party plug-in are visible in test output.

Failure modes
-------------
* Double-registering the same ``(category, name)`` raises immediately so
  duplicate decorators in two files are caught at import.
* ``build_from_config`` validates that the named class is registered and
  produces a clear error listing alternatives.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from adaptive_tutor.core.exceptions import ContractError

T = TypeVar("T")


class Registry:
    """Two-level keyed registry: ``(category, name) -> class``.

    The :func:`register` decorator is a convenience wrapper around the
    *module-level* :data:`REGISTRY` instance. Instantiate your own
    :class:`Registry` for unit tests if you need isolation::

        reg = Registry()
        @reg.register("widget", "foo")
        class Foo: ...
        assert reg.build("widget", {"name": "foo"}) is not None
    """

    def __init__(self) -> None:
        self._items: dict[str, dict[str, type]] = {}

    # ---- registration ---------------------------------------------------
    def register(self, category: str, name: str) -> Callable[[type[T]], type[T]]:
        """Decorator factory. Registers ``cls`` under ``(category, name)``."""

        def _decorate(cls: type[T]) -> type[T]:
            bucket = self._items.setdefault(category, {})
            if name in bucket:
                existing = bucket[name].__qualname__
                raise ContractError(
                    f"registry: ({category!r}, {name!r}) already bound to {existing}; "
                    f"refusing to overwrite with {cls.__qualname__}"
                )
            bucket[name] = cls
            return cls

        return _decorate

    # ---- lookup ---------------------------------------------------------
    def get(self, category: str, name: str) -> type:
        bucket = self._items.get(category)
        if not bucket or name not in bucket:
            known = sorted((self._items.get(category) or {}).keys())
            raise ContractError(
                f"registry: no entry ({category!r}, {name!r}); known names in "
                f"category {category!r}: {known}"
            )
        return bucket[name]

    def list_registered(self, category: str) -> list[str]:
        return sorted((self._items.get(category) or {}).keys())

    # ---- factory --------------------------------------------------------
    def build(self, category: str, cfg: Mapping[str, Any]) -> Any:
        """Instantiate a registered class from a config dict.

        Expected ``cfg`` shape::

            {"name": "<registered-name>", "<ctor_arg>": <value>, ...}

        The ``"name"`` key is consumed by the registry; everything else
        is forwarded to the constructor as keyword arguments. Any nested
        ``"name"`` belongs to a different category and the caller is
        responsible for resolving it first.
        """
        if "name" not in cfg:
            raise ContractError(
                f"registry.build({category!r}): config is missing the "
                f"'name' key; got keys={list(cfg.keys())}"
            )
        name = str(cfg["name"])
        cls = self.get(category, name)
        kwargs = {k: v for k, v in cfg.items() if k != "name"}
        try:
            return cls(**kwargs)
        except TypeError as e:
            raise ContractError(
                f"registry.build({category!r}, {name!r}) failed to construct "
                f"{cls.__qualname__}: {e}"
            ) from e


# ---- Module-level shared instance + convenience decorators -----------------
REGISTRY = Registry()


def register(category: str, name: str) -> Callable[[type[T]], type[T]]:
    """Convenience: register on the module-level :data:`REGISTRY`."""
    return REGISTRY.register(category, name)


def build_from_config(category: str, cfg: Mapping[str, Any]) -> Any:
    """Convenience: build from the module-level :data:`REGISTRY`."""
    return REGISTRY.build(category, cfg)
