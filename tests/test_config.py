from harness.cli import main
from harness.config import Settings, load_dotenv


def test_dotenv_does_not_override_real_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nHARNESS_MODEL="from-file"\nHARNESS_RPM=5\nEMPTY=\n')
    monkeypatch.setenv("HARNESS_MODEL", "from-env")
    monkeypatch.delenv("HARNESS_RPM", raising=False)
    load_dotenv(env)
    import os

    assert os.environ["HARNESS_MODEL"] == "from-env"
    assert os.environ["HARNESS_RPM"] == "5"
    assert "EMPTY" not in os.environ


def test_settings_build_client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HARNESS_RPD", "100")
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path / "h"))
    s = Settings.from_env()
    c = s.client("some-model")
    assert c.model == "some-model" and c.limiter.rpd == 100


def test_check_without_key_explains(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert main(["check"]) == 1
    assert "GEMINI_API_KEY" in capsys.readouterr().out
