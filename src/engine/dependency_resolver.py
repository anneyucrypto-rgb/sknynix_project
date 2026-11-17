"""Topological dependency planning and downstream invalidation."""

from __future__ import annotations

from src.engine.feature_registry import FEATURES


SOURCE_NODES = {dependency for spec in FEATURES.values() for dependency in spec.dependencies if dependency not in FEATURES}


class DependencyCycleError(ValueError):
    pass


def resolve_dependencies(targets: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    requested = (targets,) if isinstance(targets, str) else tuple(targets)
    order: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in SOURCE_NODES:
            return
        if node not in FEATURES:
            raise KeyError(f"Unknown dataset dependency: {node}")
        if node in visiting:
            raise DependencyCycleError(f"Dependency cycle includes {node}")
        if node in visited:
            return
        visiting.add(node)
        for dependency in FEATURES[node].dependencies:
            visit(dependency)
        visiting.remove(node)
        visited.add(node)
        order.append(node)

    for target in requested:
        visit(target)
    return tuple(order)


def downstream_of(changed: str) -> tuple[str, ...]:
    affected: list[str] = []
    frontier = {changed}
    while frontier:
        current = frontier.pop()
        for name, spec in FEATURES.items():
            if name not in affected and current in spec.dependencies:
                affected.append(name)
                frontier.add(name)
    return tuple(affected)
