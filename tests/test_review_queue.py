"""E4-3 / E5-1 / E5-2 · ciclo de vida de estados, cola de revisión y resolve."""

import json

from acm.config import load_profile_taxonomy, load_settings
from acm.models import AuditDecision, CardScope, ClassifiedCard
from acm.pipeline.normalizer import normalize_text
from acm.pipeline.similarity import make_fingerprint
from acm.store.registry import Registry


def _classified(front, back, *, resolved=None, unresolved=None, note_type="Basic") -> ClassifiedCard:
    return ClassifiedCard(
        front=front, back=back,
        front_normalized=normalize_text(front), back_normalized=normalize_text(back),
        scope=CardScope(), tags_resolved=resolved or [], tags_unresolved=unresolved or [],
        note_type=note_type,
        fingerprint=make_fingerprint(normalize_text(front), normalize_text(back)),
        source="claude",
    )


def _insert(reg, classified, action) -> str:
    return reg.insert(AuditDecision(card=classified, action=action, reason="x"))


def test_status_lifecycle_and_queues(tmp_path):
    reg = Registry(tmp_path / "t.db")
    rid_clean = _insert(reg, _classified("Q1", "A1"), "insert")
    rid_dup = _insert(reg, _classified("Q2", "A2"), "possible_duplicate")
    rid_amb = _insert(reg, _classified("Q3", "A3", unresolved=["_unresolved::x"]), "insert")

    review_ids = {r["id"] for r in reg.list_pending_review()}
    sync_ids = {r["id"] for r in reg.list_pending_sync()}

    # clean → aprobada → sincronizable, no en cola de revisión
    assert reg.get_by_id(rid_clean)["status"] == "aprobada"
    assert rid_clean in sync_ids and rid_clean not in review_ids
    # duplicado → en-revision
    assert reg.get_by_id(rid_dup)["status"] == "en-revision"
    assert rid_dup in review_ids and rid_dup not in sync_ids
    # E5-1: ambiguo (tags sin resolver) → en-revision, NO auto-sincronizable
    assert reg.get_by_id(rid_amb)["status"] == "en-revision"
    assert rid_amb in review_ids and rid_amb not in sync_ids


def test_approve_makes_ambiguous_syncable(tmp_path):
    reg = Registry(tmp_path / "t.db")
    rid = _insert(reg, _classified("Q", "A", unresolved=["_unresolved::x"]), "insert")
    assert rid not in {r["id"] for r in reg.list_pending_sync()}
    reg.update_action(rid, "insert")  # approve → aprobada
    assert reg.get_by_id(rid)["status"] == "aprobada"
    assert rid in {r["id"] for r in reg.list_pending_sync()}


def test_uploaded_is_idempotent(tmp_path):
    """E4-3: una vez subida (anki_note_id), no vuelve a la cola de sync."""
    reg = Registry(tmp_path / "t.db")
    rid = _insert(reg, _classified("Q", "A"), "insert")
    assert rid in {r["id"] for r in reg.list_pending_sync()}
    reg.update_action(rid, "insert", anki_note_id=12345)
    assert reg.get_by_id(rid)["status"] == "subida"
    assert rid not in {r["id"] for r in reg.list_pending_sync()}


def test_reject_discards(tmp_path):
    reg = Registry(tmp_path / "t.db")
    rid = _insert(reg, _classified("Q", "A"), "possible_duplicate")
    reg.update_action(rid, "reject")
    assert reg.get_by_id(rid)["status"] == "descartada"
    assert rid not in {r["id"] for r in reg.list_pending_review()}


def _mock_setup(reg):
    settings = load_settings()
    return lambda *a, **k: (reg, settings, "default", None, None)


def test_acm_resolve_approve(tmp_path, monkeypatch):
    import acm.mcp_server as srv
    reg = Registry(tmp_path / "t.db")
    rid = _insert(reg, _classified("Q", "A"), "possible_duplicate")
    monkeypatch.setattr(srv, "_setup", _mock_setup(reg))
    res = json.loads(srv.acm_resolve(rid, "approve"))
    assert res["status"] == "approved"
    assert res["estado"] == "aprobada"
    assert reg.get_by_id(rid)["status"] == "aprobada"


def test_acm_resolve_reject_by_prefix(tmp_path, monkeypatch):
    import acm.mcp_server as srv
    reg = Registry(tmp_path / "t.db")
    rid = _insert(reg, _classified("Q", "A"), "possible_duplicate")
    monkeypatch.setattr(srv, "_setup", _mock_setup(reg))
    res = json.loads(srv.acm_resolve(rid[:8], "reject"))
    assert res["status"] == "rejected"
    assert reg.get_by_id(rid)["status"] == "descartada"


def test_acm_resolve_invalid_action(tmp_path, monkeypatch):
    import acm.mcp_server as srv
    reg = Registry(tmp_path / "t.db")
    monkeypatch.setattr(srv, "_setup", _mock_setup(reg))
    res = json.loads(srv.acm_resolve("whatever", "frobnicate"))
    assert "error" in res


def test_reject_works_on_approved_record_by_prefix(tmp_path):
    """§6: rechazar por PREFIJO una card ya aprobada (fuera de la cola) funciona."""
    import acm.service as service
    reg = Registry(tmp_path / "t.db")
    rid = _insert(reg, _classified("Q", "A"), "insert")  # aprobada (no en-revision)
    res = service.resolve_record(reg, rid[:8], "reject")
    assert res["status"] == "rejected"
    assert reg.get_by_id(rid)["status"] == "descartada"


def test_acm_ingest_is_idempotent_on_reingest(tmp_path, monkeypatch):
    """§7: re-ingerir el mismo contenido actualiza en sitio, no duplica."""
    import acm.mcp_server as srv

    settings = load_settings()
    settings.acm.use_embeddings = False
    name, profile, taxonomy = load_profile_taxonomy(settings)
    reg = Registry(tmp_path / "t.db")
    monkeypatch.setattr(srv, "_setup", lambda *a, **k: (reg, settings, name, profile, taxonomy))

    class _NoAnki:
        def __init__(self, *a, **k):
            pass

        def is_available(self):
            return False

    monkeypatch.setattr(srv, "AnkiConnectClient", _NoAnki)

    cards = json.dumps([{"front": "What is Azure Cloud Shell?", "back": "A browser shell", "source": "claude"}])
    srv.acm_ingest(cards)
    second = json.loads(srv.acm_ingest(cards))  # re-ingesta idéntica
    assert len(reg.list_processed_cards()) == 1  # no se duplicó el registro
    assert second["summary"].get("updated") == 1


def test_correct_can_change_note_type(tmp_path, monkeypatch):
    """§5: corregir puede cambiar el note_type (no solo front/back/tags)."""
    import acm.mcp_server as srv

    settings = load_settings()
    settings.acm.use_embeddings = False
    name, profile, taxonomy = load_profile_taxonomy(settings)
    reg = Registry(tmp_path / "t.db")
    rid = _insert(reg, _classified("Q", "A", note_type="Basic"), "possible_duplicate")
    monkeypatch.setattr(srv, "_setup", lambda *a, **k: (reg, settings, name, profile, taxonomy))
    monkeypatch.setattr(srv, "_try_anki_client", lambda s: None)

    res = json.loads(srv.acm_resolve(rid, "correct", front="What is X?", back="Y",
                                     note_type="Cloze EduSksh"))
    assert res["status"] == "corrected"
    assert reg.get_by_id(rid)["note_type"] == "Cloze EduSksh"


class _FakeAnki:
    def __init__(self, models=("Básico",)):
        self.added = []
        self.deleted = []
        self._models = list(models)

    def is_available(self):
        return True

    def get_model_names(self):
        return self._models

    def resolve_deck(self, **kwargs):
        return "Cloud Certs"

    def add_note(self, deck, model, fields, tags):
        self.added.append({"deck": deck, "model": model, "fields": fields})
        return 1000 + len(self.added)

    def delete_notes(self, note_ids):
        self.deleted.extend(note_ids)

    def close(self):
        pass


def test_sync_resolves_basic_to_basico(tmp_path):
    """Reporte §4: note_type 'Basic' → modelo real 'Básico' vía alias/default."""
    import acm.service as service
    reg = Registry(tmp_path / "t.db")
    _insert(reg, _classified("Q", "A"), "insert")  # note_type='Basic'
    fake = _FakeAnki(models=["Básico"])
    res = service.sync_pending(reg, load_settings(), anki_client=fake, backup=False)
    assert res["synced_count"] == 1
    assert fake.added[0]["model"] == "Básico"


def test_sync_preflight_aborts_on_unknown_model(tmp_path):
    """Reporte §4: modelo inexistente → preflight aborta SIN subidas parciales."""
    import acm.service as service
    reg = Registry(tmp_path / "t.db")
    _insert(reg, _classified("Q", "A", note_type="Cloze NoExiste"), "insert")
    fake = _FakeAnki(models=["Básico"])
    res = service.sync_pending(reg, load_settings(), anki_client=fake, backup=False)
    assert "error" in res
    assert res["problems"][0]["note_type"] == "Cloze NoExiste"
    assert fake.added == []  # nada subido
    assert len(reg.list_pending_sync()) == 1  # sigue encolada (no se perdió)


def test_resolve_model_name_paths():
    from acm.config import load_settings
    from acm.service import resolve_model_name
    settings = load_settings()  # model_aliases: Basic→Básico, Cloze→Cloze EduSksh
    models = ["Básico", "Cloze EduSksh"]
    assert resolve_model_name("Básico", settings, models) == ("Básico", None)
    assert resolve_model_name("Basic", settings, models)[0] == "Básico"   # alias
    assert resolve_model_name("Cloze", settings, models)[0] == "Cloze EduSksh"  # alias
    model, error = resolve_model_name("Basicoo", settings, models)  # typo
    assert model is None and "Básico" in error  # sugerencia por cercanía


def test_sync_dry_run_does_not_upload(tmp_path):
    import acm.service as service
    reg = Registry(tmp_path / "t.db")
    rid = _insert(reg, _classified("Q", "A"), "insert")
    fake = _FakeAnki()
    res = service.sync_pending(reg, load_settings(), anki_client=fake, dry_run=True)
    assert res["dry_run"] is True
    assert res["count"] == 1
    assert fake.added == []  # E9-1: nada subido
    assert reg.get_by_id(rid)["status"] == "aprobada"


def test_backup_registry_creates_copy(tmp_path, monkeypatch):
    import acm.service as service
    monkeypatch.setenv("ACM_HOME", str(tmp_path / "home"))
    settings = load_settings()
    Registry(settings.db_path_resolved)  # crea ACM_HOME/registry.db
    backup = service.backup_registry(settings)
    assert backup.exists()
    assert backup.parent.name == "backups"


def test_sync_and_undo_batch(tmp_path):
    import acm.service as service
    reg = Registry(tmp_path / "t.db")
    rid = _insert(reg, _classified("Q", "A"), "insert")
    fake = _FakeAnki()

    res = service.sync_pending(reg, load_settings(), anki_client=fake, backup=False)
    assert res["synced_count"] == 1
    note_id = res["synced"][0]["note_id"]
    assert reg.get_by_id(rid)["status"] == "subida"
    assert res["batch_id"]

    undo = service.undo_batch(reg, load_settings(), res["batch_id"], anki_client=fake)
    assert undo["deleted_notes"] == 1
    assert note_id in fake.deleted
    assert reg.get_by_id(rid)["status"] == "aprobada"  # E9-2: revertido
    assert reg.get_by_id(rid)["anki_note_id"] is None


def test_correct_reingests_and_reclassifies(tmp_path, monkeypatch):
    """E5-3: corregir una card la re-clasifica y re-deduplica."""
    import acm.mcp_server as srv
    from acm.config import load_profile_taxonomy
    settings = load_settings()
    settings.acm.use_embeddings = False
    name, profile, taxonomy = load_profile_taxonomy(settings)
    reg = Registry(tmp_path / "t.db")
    rid = _insert(reg, _classified("vague q", "vague a", unresolved=["_unresolved::x"]), "insert")
    assert reg.get_by_id(rid)["status"] == "en-revision"

    monkeypatch.setattr(srv, "_setup", lambda *a, **k: (reg, settings, name, profile, taxonomy))
    monkeypatch.setattr(srv, "_try_anki_client", lambda s: None)

    res = json.loads(srv.acm_resolve(
        rid, "correct", front="What is Azure AD?", back="Cloud identity service",
        tags=["vendor::azure", "topic::identity", "type::definition"],
    ))
    assert res["status"] == "corrected"
    assert "vendor::azure" in res["tags_resolved"]
    # contenido y clasificación actualizados en el registro
    row = reg.get_by_id(rid)
    assert row["front_original"] == "What is Azure AD?"
    assert "vendor::azure" in row["tags_resolved"]
    assert row["status"] == "aprobada"  # ya sin ambigüedad


def test_metrics_auto_vs_escalated(tmp_path):
    reg = Registry(tmp_path / "t.db")
    _insert(reg, _classified("Q1", "A1"), "insert")  # aprobada
    _insert(reg, _classified("Q2", "A2"), "possible_duplicate")  # en-revision
    m = reg.metrics()
    assert m["auto_resolved"] == 1
    assert m["escalated_to_user"] == 1


def test_sync_offline_queues_and_exports_tsv(tmp_path, monkeypatch):
    """E4-5: con Anki cerrado, las aprobadas quedan encoladas y se exportan a TSV."""
    import acm.service as service
    reg = Registry(tmp_path / "t.db")
    _insert(reg, _classified("What is a VPC?", "A virtual network", resolved=["topic::networking"]), "insert")
    monkeypatch.setattr(service, "try_anki_client", lambda settings: None)  # Anki caído

    tsv = tmp_path / "fallback.tsv"
    result = service.sync_pending(reg, load_settings(), export_tsv_path=tsv)

    assert result["anki_available"] is False
    assert result["queued"] == 1
    assert result["exported_count"] == 1
    assert tsv.exists()
    assert "What is a VPC?" in tsv.read_text(encoding="utf-8")
    # La card sigue aprobada (encolada), no subida.
    assert reg.get_by_id(reg.list_pending_sync()[0]["id"])["status"] == "aprobada"
