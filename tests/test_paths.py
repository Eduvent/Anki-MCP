"""E0-1: rutas absolutas estables vía ACM_HOME, independientes del cwd."""

from acm.config import Settings, acm_home, load_settings


def test_db_path_is_absolute_and_under_acm_home(tmp_path, monkeypatch):
    monkeypatch.setenv("ACM_HOME", str(tmp_path / "home"))
    settings = Settings()  # default db_path="registry.db"
    resolved = settings.db_path_resolved
    assert resolved.is_absolute()
    assert resolved == (tmp_path / "home" / "registry.db").resolve()


def test_same_db_across_different_cwds(tmp_path, monkeypatch):
    """El núcleo de E0-1: lanzado desde cwd distintos, mismo DB."""
    monkeypatch.setenv("ACM_HOME", str(tmp_path / "home"))
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    monkeypatch.chdir(dir_a)
    path_a = load_settings().db_path_resolved
    monkeypatch.chdir(dir_b)
    path_b = load_settings().db_path_resolved

    assert path_a == path_b
    assert path_a.is_absolute()


def test_absolute_db_path_is_respected(tmp_path, monkeypatch):
    monkeypatch.setenv("ACM_HOME", str(tmp_path / "home"))
    abs_db = tmp_path / "custom" / "my.db"
    settings = Settings(acm={"db_path": str(abs_db)})
    assert settings.db_path_resolved == abs_db


def test_acm_home_env_override(tmp_path, monkeypatch):
    target = tmp_path / "elsewhere"
    monkeypatch.setenv("ACM_HOME", str(target))
    assert acm_home() == target.resolve()
    assert target.exists()


def test_taxonomy_empty_defaults_to_packaged(tmp_path, monkeypatch):
    monkeypatch.setenv("ACM_HOME", str(tmp_path / "home"))
    settings = Settings()  # taxonomy_path=""
    resolved = settings.taxonomy_path_resolved
    assert resolved.is_absolute()
    assert resolved.name == "taxonomy.yaml"
    assert resolved.exists()
