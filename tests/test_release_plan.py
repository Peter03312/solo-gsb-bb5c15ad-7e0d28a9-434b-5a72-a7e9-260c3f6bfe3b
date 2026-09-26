"""API tests for POST /release-plans/preview (read-only orchestration)."""

import os
import tempfile

os.environ["DATABASE_PATH"] = os.path.join(tempfile.mkdtemp(), "test.db")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def resolved_merge(client, base=1, value=2):
    """A conflict-free merge order, immediately resolved at revision 1."""
    resp = client.post(
        "/merges",
        json={"base": {"v": base}, "left": {"v": value}, "right": {"v": value}},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["status"] == "resolved"
    return body["merge_id"]


def pending_merge(client):
    """A merge order with one open conflict, staying pending."""
    resp = client.post(
        "/merges",
        json={
            "base": {"a": 1},
            "left": {"a": 2},
            "right": {"a": 3},
        },
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["status"] == "pending"
    return body["merge_id"]


def resolve_conflict(client, merge_id):
    resp = client.post(
        f"/merges/{merge_id}/resolutions",
        json={"revision": 1, "resolutions": [{"pointer": "/a", "choice": "left"}]},
    )
    assert resp.status_code == 200, resp.text


def preview(client, nodes):
    return client.post("/release-plans/preview", json={"nodes": nodes})


def n(merge_id, *depends_on):
    return {"merge_id": merge_id, "depends_on": list(depends_on)}


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_all_resolved_no_edges_single_batch_sorted(client):
    a = resolved_merge(client, 1, 11)
    b = resolved_merge(client, 2, 22)

    resp = preview(client, [n(b), n(a)])  # request order intentionally reversed
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["batches"] == [{"batch": 1, "merge_ids": sorted([a, b])}]
    assert body["blocked"] == []
    assert {node["merge_id"] for node in body["nodes"]} == {a, b}
    assert all(node["revision"] == 1 for node in body["nodes"])
    assert all(node["status"] == "resolved" for node in body["nodes"])
    assert all(node["depends_on"] == [] for node in body["nodes"])


def test_batches_follow_dependency_layers(client):
    # c -> b -> a and c -> a: a first, then b, then c.
    a = resolved_merge(client)
    b = resolved_merge(client)
    c = resolved_merge(client)

    resp = preview(client, [n(c, b, a), n(b, a), n(a)])
    assert resp.status_code == 200, resp.text
    assert resp.json()["batches"] == [
        {"batch": 1, "merge_ids": [a]},
        {"batch": 2, "merge_ids": [b]},
        {"batch": 3, "merge_ids": [c]},
    ]


def test_diamond_graph_layers_independent_nodes_together(client):
    # top -> {mid1, mid2} -> base
    base = resolved_merge(client)
    mid1 = resolved_merge(client)
    mid2 = resolved_merge(client)
    top = resolved_merge(client)

    resp = preview(
        client,
        [n(top, mid1, mid2), n(mid1, base), n(mid2, base), n(base)],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["batches"] == [
        {"batch": 1, "merge_ids": [base]},
        {"batch": 2, "merge_ids": sorted([mid1, mid2])},
        {"batch": 3, "merge_ids": [top]},
    ]
    assert body["blocked"] == []


def test_pending_node_is_self_pending(client):
    a = pending_merge(client)

    resp = preview(client, [n(a)])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["batches"] == []
    assert body["blocked"] == [
        {"merge_id": a, "reason": "self_pending", "blocked_by": None}
    ]
    node = body["nodes"][0]
    assert node["merge_id"] == a
    assert node["status"] == "pending"
    assert node["revision"] == 1


def test_pending_node_blocks_successors_with_direct_dependency(client):
    # pending p blocks resolved r (depends on p); r in turn blocks s.
    p = pending_merge(client)
    r = resolved_merge(client)
    s = resolved_merge(client)

    resp = preview(client, [n(s, r), n(r, p), n(p)])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["batches"] == []
    # Blocked nodes are reported in stable merge_id order; compare by id.
    by_id = {b["merge_id"]: b for b in body["blocked"]}
    assert [b["merge_id"] for b in body["blocked"]] == sorted([p, r, s])
    assert by_id[p] == {"merge_id": p, "reason": "self_pending", "blocked_by": None}
    assert by_id[r] == {
        "merge_id": r,
        "reason": "blocked_by_dependency",
        "blocked_by": p,
    }
    assert by_id[s] == {
        "merge_id": s,
        "reason": "blocked_by_dependency",
        "blocked_by": r,
    }


def test_partial_plan_publishes_unaffected_branches(client):
    # pending p with resolved follower f stays blocked; independent
    # resolved chain base -> top still gets batches.
    p = pending_merge(client)
    f = resolved_merge(client)
    base = resolved_merge(client)
    top = resolved_merge(client)

    resp = preview(client, [n(f, p), n(p), n(top, base), n(base)])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["batches"] == [
        {"batch": 1, "merge_ids": [base]},
        {"batch": 2, "merge_ids": [top]},
    ]
    assert {b["merge_id"] for b in body["blocked"]} == {p, f}
    by_id = {b["merge_id"]: b for b in body["blocked"]}
    assert by_id[p]["reason"] == "self_pending"
    assert by_id[f] == {
        "merge_id": f,
        "reason": "blocked_by_dependency",
        "blocked_by": p,
    }


def test_resolved_pending_node_then_publishable(client):
    p = pending_merge(client)
    r = resolved_merge(client)
    req_nodes = [n(r, p), n(p)]

    before = preview(client, req_nodes).json()
    assert before["batches"] == []
    assert {b["merge_id"] for b in before["blocked"]} == {p, r}

    resolve_conflict(client, p)

    after = preview(client, req_nodes)
    assert after.status_code == 200, after.text
    body = after.json()
    assert body["batches"] == [
        {"batch": 1, "merge_ids": [p]},
        {"batch": 2, "merge_ids": [r]},
    ]
    assert body["blocked"] == []
    # The formerly pending node's revision moved; the response lets the
    # caller see exactly which node changed.
    revisions = {node["merge_id"]: node["revision"] for node in body["nodes"]}
    assert revisions[p] == 2
    assert revisions[r] == 1


def test_pending_node_with_its_own_blocked_prerequisite_is_self_pending(client):
    p = pending_merge(client)
    x = pending_merge(client)
    top = resolved_merge(client)
    # `x` is pending and blocked behind `p`; its own pending state is what
    # is reported, not the upstream dependency.
    resp = preview(client, [n(top, x), n(x, p), n(p)])
    assert resp.status_code == 200, resp.text
    by_id = {b["merge_id"]: b for b in resp.json()["blocked"]}
    assert by_id[p]["reason"] == "self_pending"
    assert by_id[x]["reason"] == "self_pending"
    assert by_id[x]["blocked_by"] is None
    # The resolved node names the lexicographically smallest blocked
    # prerequisite when several exist.
    assert by_id[top]["reason"] == "blocked_by_dependency"
    assert by_id[top]["blocked_by"] == x


def test_blocked_by_names_smallest_direct_prerequisite(client):
    p1 = pending_merge(client)
    p2 = pending_merge(client)
    r = resolved_merge(client)
    resp = preview(client, [n(r, p2, p1), n(p1), n(p2)])
    assert resp.status_code == 200, resp.text
    by_id = {b["merge_id"]: b for b in resp.json()["blocked"]}
    assert by_id[r]["blocked_by"] == min(p1, p2)


def test_revisions_are_echoed_per_node(client):
    p = pending_merge(client)
    resolve_conflict(client, p)
    r = resolved_merge(client)

    body = preview(client, [n(r, p), n(p)]).json()
    revisions = {node["merge_id"]: node for node in body["nodes"]}
    assert revisions[p]["revision"] == 2
    assert revisions[p]["status"] == "resolved"
    assert revisions[r]["revision"] == 1


def test_preview_is_repeatable_and_writes_nothing(client):
    from app import db

    p = pending_merge(client)
    r = resolved_merge(client)
    payload = {"nodes": [n(r, p), n(p)]}
    first = client.post("/release-plans/preview", json=payload)
    second = client.post("/release-plans/preview", json=payload)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()

    count_before = db.query_one(
        "SELECT COUNT(*) AS n FROM merge_audit_events"
    )["n"]
    client.post("/release-plans/preview", json=payload)
    count_after = db.query_one(
        "SELECT COUNT(*) AS n FROM merge_audit_events"
    )["n"]
    assert count_after == count_before


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------


def test_duplicate_merge_id_rejected(client):
    a = resolved_merge(client)
    resp = preview(client, [n(a), n(a)])
    assert resp.status_code == 400
    body = resp.json()
    assert body["reason"] == "duplicate_node"
    assert body["merge_id"] == a


def test_self_dependency_rejected(client):
    a = resolved_merge(client)
    resp = preview(client, [n(a, a)])
    assert resp.status_code == 400
    body = resp.json()
    assert body["reason"] == "self_dependency"
    assert body["merge_id"] == a


def test_dependency_not_in_list_rejected(client):
    a = resolved_merge(client)
    ghost = "0" * 32
    resp = preview(client, [n(a, ghost)])
    assert resp.status_code == 400
    body = resp.json()
    assert body["reason"] == "unknown_dependency"
    assert body["merge_id"] == a
    assert body["missing_dependency"] == ghost


def test_cycle_rejected(client):
    a = resolved_merge(client)
    b = resolved_merge(client)
    c = resolved_merge(client)
    resp = preview(client, [n(a, b), n(b, c), n(c, a)])
    assert resp.status_code == 400
    body = resp.json()
    assert body["reason"] == "cycle_detected"
    assert sorted(body["cycle"]) == sorted([a, b, c])


def test_two_node_cycle_rejected(client):
    a = resolved_merge(client)
    b = resolved_merge(client)
    resp = preview(client, [n(a, b), n(b, a)])
    assert resp.status_code == 400
    assert resp.json()["reason"] == "cycle_detected"


def test_unknown_merge_rejects_whole_plan(client):
    a = resolved_merge(client)
    ghost = "f" * 32
    # Even though `a` exists and is publishable, the unknown node fails
    # the entire request.
    resp = preview(client, [n(a), n(ghost)])
    assert resp.status_code == 400
    body = resp.json()
    assert body["reason"] == "unknown_merge"
    assert body["merge_id"] == ghost


def test_empty_nodes_rejected(client):
    resp = client.post("/release-plans/preview", json={"nodes": []})
    assert resp.status_code == 422


def test_malformed_node_rejected(client):
    resp = client.post(
        "/release-plans/preview", json={"nodes": [{"depends_on": []}]}
    )
    assert resp.status_code == 422


def test_depends_on_must_be_a_list(client):
    a = resolved_merge(client)
    resp = client.post(
        "/release-plans/preview",
        json={"nodes": [{"merge_id": a, "depends_on": "not-a-list"}]},
    )
    assert resp.status_code == 422


def test_empty_merge_id_rejected(client):
    resp = client.post(
        "/release-plans/preview",
        json={"nodes": [{"merge_id": "", "depends_on": []}]},
    )
    assert resp.status_code == 422


def test_duplicate_dependency_edges_are_deduplicated(client):
    a = resolved_merge(client)
    b = resolved_merge(client)
    resp = preview(client, [{"merge_id": b, "depends_on": [a, a]}, n(a)])
    assert resp.status_code == 200, resp.text
    node_b = next(node for node in resp.json()["nodes"] if node["merge_id"] == b)
    assert node_b["depends_on"] == [a]
    assert resp.json()["batches"] == [
        {"batch": 1, "merge_ids": [a]},
        {"batch": 2, "merge_ids": [b]},
    ]


def test_long_chain_beyond_recursion_limit(client):
    # Well over sys.getrecursionlimit(): validation and batching must not
    # rely on recursive graph walks. Conflict-free merges are created via
    # the API (each persists a row); build the chain afterwards in the
    # request payload.
    import sys

    size = sys.getrecursionlimit() + 500
    ids = [resolved_merge(client, value=i) for i in range(size)]
    nodes = [n(ids[0])] + [
        n(ids[i], ids[i - 1]) for i in range(1, size)
    ]

    resp = preview(client, nodes)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["blocked"] == []
    assert [batch["batch"] for batch in body["batches"]] == list(
        range(1, size + 1)
    )
    assert body["batches"][0]["merge_ids"] == [ids[0]]
    assert body["batches"][-1]["merge_ids"] == [ids[-1]]

    # The same long chain with one pending node in the middle: everything
    # from that node onward is blocked, no recursion error either.
    from app import db

    db.execute("UPDATE merges SET status = 'pending' WHERE id = ?", (ids[100],))
    body = preview(client, nodes).json()
    assert len(body["batches"]) == 100
    blocked_ids = {b["merge_id"] for b in body["blocked"]}
    assert blocked_ids == set(ids[100:])
