"""Release-plan orchestration preview: dependency-graph validation and batching.

Pure graph logic: no SQLite access here. The caller (``store``) supplies each
node's current merge status after the graph itself has been validated, so the
planning functions are deterministic and side-effect free. Nothing here is
ever persisted — a release plan is computed from the *current* state of the
referenced merge orders.
"""

from __future__ import annotations

from typing import Any

# A node is blocked because its own merge order is not resolved yet.
REASON_SELF_PENDING = "self_pending"
# A resolved node is blocked by a directly referenced dependency that cannot
# itself be published.
REASON_BLOCKED_BY_DEPENDENCY = "blocked_by_dependency"


class InvalidReleaseGraph(Exception):
    """Base class: the supplied node list does not describe a valid plan."""

    reason = "invalid_release_graph"

    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(detail)


class DuplicateNode(InvalidReleaseGraph):
    reason = "duplicate_node"

    def __init__(self, merge_id: str):
        self.merge_id = merge_id
        super().__init__(f"merge_id {merge_id!r} appears more than once")


class SelfDependency(InvalidReleaseGraph):
    reason = "self_dependency"

    def __init__(self, merge_id: str):
        self.merge_id = merge_id
        super().__init__(f"node {merge_id!r} depends on itself")


class UnknownDependency(InvalidReleaseGraph):
    reason = "unknown_dependency"

    def __init__(self, merge_id: str, missing: str):
        self.merge_id = merge_id
        self.missing = missing
        super().__init__(
            f"node {merge_id!r} depends on {missing!r}, which is not in the node list"
        )


class CyclicReleaseGraph(InvalidReleaseGraph):
    reason = "cycle_detected"

    def __init__(self, nodes: list[str]):
        # Sorted for a stable message regardless of the request order.
        self.nodes = sorted(nodes)
        super().__init__(
            "dependency cycle involving: " + ", ".join(self.nodes)
        )


class UnknownPlanMerge(InvalidReleaseGraph):
    """A referenced node has no merge order in the database."""

    reason = "unknown_merge"

    def __init__(self, merge_id: str):
        self.merge_id = merge_id
        super().__init__(f"merge {merge_id!r} does not exist")


def normalize_graph(nodes: list[dict]) -> dict[str, list[str]]:
    """Validate the request graph; return ``merge_id -> sorted dependencies``.

    ``nodes`` is a list of ``{"merge_id": ..., "depends_on": [...]}`` dicts.
    Checks, in order:

    1. ``merge_id`` values are unique;
    2. no node lists itself as a dependency;
    3. every dependency refers to another node in the list;
    4. the graph is acyclic.

    Each failure raises the matching InvalidReleaseGraph subclass, so the
    whole request is rejected before any merge state is read.
    """
    seen: set[str] = set()
    for node in nodes:
        merge_id = node["merge_id"]
        if merge_id in seen:
            raise DuplicateNode(merge_id)
        seen.add(merge_id)

    graph: dict[str, list[str]] = {}
    for node in nodes:
        merge_id = node["merge_id"]
        depends_on: list[str] = []
        for dep in node.get("depends_on") or []:
            if dep == merge_id:
                raise SelfDependency(merge_id)
            if dep not in seen:
                raise UnknownDependency(merge_id, dep)
            depends_on.append(dep)
        # Deduplicate repeated edges and normalize order; whether the
        # request lists a dependency once or twice never affects batches.
        graph[merge_id] = sorted(set(depends_on))

    cycle = _find_cycle(graph)
    if cycle:
        raise CyclicReleaseGraph(cycle)
    return graph


def _find_cycle(graph: dict[str, list[str]]) -> list[str] | None:
    """Return the nodes of one cycle (via a DFS back edge), or None.

    Iterative DFS with an explicit stack, so dependency chains longer than
    the Python recursion limit are handled. ``graph`` is already normalized
    and self-loops are rejected before this runs.
    """
    color: dict[str, int] = {}  # 1 = on the current path (gray), 2 = done
    for start in sorted(graph):
        if color.get(start, 0) != 0:
            continue
        color[start] = 1
        path: list[str] = [start]
        # (node, index of the next neighbor to visit).
        stack: list[tuple[str, int]] = [(start, 0)]
        while stack:
            v, i = stack[-1]
            if i < len(graph[v]):
                stack[-1] = (v, i + 1)
                w = graph[v][i]
                state = color.get(w, 0)
                if state == 0:
                    color[w] = 1
                    path.append(w)
                    stack.append((w, 0))
                elif state == 1:
                    # Back edge to a node on the current path: the path
                    # segment from w to v closes through the edge v -> w.
                    return path[path.index(w):]
            else:
                color[v] = 2
                path.pop()
                stack.pop()
    return None


def build_release_plan(
    graph: dict[str, list[str]], merges: dict[str, dict[str, Any]]
) -> dict:
    """Compute stable release batches from current merge states.

    ``merges`` maps every graph node to its current view (at least
    ``status`` and ``revision``). A node is *publishable* only when its own
    merge order is ``resolved`` and every dependency is publishable too.

    Publishable nodes are layered by longest dependency path and emitted as
    batches: batch 1 has no prerequisites, each later batch depends only on
    earlier batches, and nodes within a batch are sorted by ``merge_id``.
    Every non-publishable node is reported in ``blocked`` with the reason
    ``self_pending`` for an unresolved merge order or
    ``blocked_by_dependency`` naming the lexicographically smallest direct
    dependency that cannot be published.

    The graph is assumed acyclic (``normalize_graph`` rejects cycles).
    Propagation is iterative, so plans with long dependency chains are not
    limited by the Python recursion depth.
    """
    resolved = {
        merge_id
        for merge_id, view in merges.items()
        if view["status"] == "resolved"
    }

    # Publishability propagates bottom-up: a resolved node whose (publishable
    # or resolved) prerequisites have all been confirmed publishable enters
    # the queue; a pending prerequisite permanently holds its followers.
    dependents: dict[str, list[str]] = {merge_id: [] for merge_id in graph}
    remaining_deps: dict[str, int] = {merge_id: 0 for merge_id in graph}
    for merge_id, deps in graph.items():
        remaining_deps[merge_id] = len(deps)
        for dep in deps:
            dependents[dep].append(merge_id)

    # Seed with resolved nodes that have no prerequisites at all.
    queue: list[str] = sorted(
        merge_id
        for merge_id in graph
        if merge_id in resolved and not graph[merge_id]
    )
    publishable: set[str] = set()
    head = 0
    while head < len(queue):
        merge_id = queue[head]
        head += 1
        if merge_id not in publishable:
            publishable.add(merge_id)
        for follower in dependents[merge_id]:
            remaining_deps[follower] -= 1
            if follower in resolved and remaining_deps[follower] == 0:
                # Every prerequisite was published in turn (pending nodes
                # never enter the queue), so the follower is publishable.
                queue.append(follower)

    # Layer number = 1 + the maximum layer of the prerequisites, computed in
    # the bottom-up publication order, which is itself a topological order.
    layer: dict[str, int] = {}
    for merge_id in queue:
        layer[merge_id] = (
            1 + max((layer[dep] for dep in graph[merge_id]), default=0)
        )

    blocked: list[dict[str, Any]] = []
    for merge_id in sorted(graph):
        if merge_id in publishable:
            continue
        if merge_id not in resolved:
            blocked.append(
                {
                    "merge_id": merge_id,
                    "reason": REASON_SELF_PENDING,
                    "blocked_by": None,
                }
            )
        else:
            # A resolved node is reachable here only because at least one
            # direct dependency is not publishable.
            blocker = min(
                dep for dep in graph[merge_id] if dep not in publishable
            )
            blocked.append(
                {
                    "merge_id": merge_id,
                    "reason": REASON_BLOCKED_BY_DEPENDENCY,
                    "blocked_by": blocker,
                }
            )

    batches_by_layer: dict[int, list[str]] = {}
    for merge_id in layer:
        batches_by_layer.setdefault(layer[merge_id], []).append(merge_id)

    batches = [
        {"batch": batch_no, "merge_ids": sorted(batches_by_layer[level])}
        for batch_no, level in enumerate(sorted(batches_by_layer), start=1)
    ]

    # Per-node view: current status, revision and the normalized edge list,
    # so a caller can tell whether the plan was computed against a state it
    # still considers current.
    node_views = [
        {
            "merge_id": merge_id,
            "status": merges[merge_id]["status"],
            "revision": merges[merge_id]["revision"],
            "depends_on": graph[merge_id],
        }
        for merge_id in sorted(graph)
    ]

    return {
        "batches": batches,
        "blocked": blocked,
        "nodes": node_views,
    }
