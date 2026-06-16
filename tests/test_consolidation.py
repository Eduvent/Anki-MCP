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


# §9/§10: economía de tokens en la salida + gate de mejor_version

def _full_match(suggestion="keep_existing"):
    return {
        "id": "anki:1", "deck": "Cloud Certs::AWS", "score": 0.95,
        "reason_codes": ["exact_fingerprint", "token_overlap"],
        "front": "una pregunta larga " * 10, "back_excerpt": "...", "scope": {"vendor": "aws"},
        "mejor_version": {"suggestion": suggestion, "reason": "x"},
    }


def test_compact_match_drops_bulk_and_gates_mejor_version():
    compact = srv._compact_match(_full_match("keep_existing"))
    assert set(compact.keys()) == {"id", "deck", "score", "reason_codes"}  # sin front/scope/back
    assert "mejor_version" not in compact  # §10: keep_existing no se emite
    replace = srv._compact_match(_full_match("replace_old_with_new"))
    assert replace["mejor_version"]["suggestion"] == "replace_old_with_new"  # accionable sí


def test_matches_out_caps_and_respects_verbose():
    full = [_full_match(), _full_match(), _full_match()]
    assert srv._matches_out(full, verbose=True) == full  # detalle completo
    compact = srv._matches_out(full, verbose=False, limit=2)
    assert len(compact) == 2 and "front" not in compact[0]  # top-2 compacto
