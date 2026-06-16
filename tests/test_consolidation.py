"""E6-2 · consolidación de tools: acm_audit(mode) y acm_taxonomy(action)."""

import json

import acm.mcp_server as srv


def test_acm_audit_invalid_mode():
    res = json.loads(srv.acm_audit("Deck", mode="bogus"))
    assert "error" in res


def test_acm_audit_dispatches_by_mode(monkeypatch):
    monkeypatch.setattr(srv, "acm_find_duplicates", lambda deck, **k: json.dumps({"called": "dup"}))
    monkeypatch.setattr(srv, "acm_audit_recent", lambda deck, **k: json.dumps({"called": "recent"}))
    monkeypatch.setattr(srv, "acm_list_untagged", lambda deck, **k: json.dumps({"called": "untagged"}))

    assert json.loads(srv.acm_audit("D", mode="duplicates"))["called"] == "dup"
    assert json.loads(srv.acm_audit("D", mode="recent"))["called"] == "recent"
    assert json.loads(srv.acm_audit("D", mode="untagged"))["called"] == "untagged"


def test_acm_taxonomy_show():
    res = json.loads(srv.acm_taxonomy(action="show"))
    assert isinstance(res, dict)
    assert "vendor" in res  # taxonomía empaquetada


def test_acm_taxonomy_invalid_action():
    res = json.loads(srv.acm_taxonomy(action="bogus"))
    assert "error" in res


def test_acm_taxonomy_add_existing_does_not_write():
    # 'aws' ya existe en vendor → 'exists' sin escribir el archivo.
    res = json.loads(srv.acm_taxonomy(action="add", category="vendor", value="aws"))
    assert res["status"] == "exists"


def test_acm_taxonomy_add_requires_category_and_value():
    res = json.loads(srv.acm_taxonomy(action="add"))
    assert "error" in res
