"""
Unit tests for tools/fork.py.
"""
import sys
from pathlib import Path

import pytest
import yaml

# Add tools/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from fork import fork_model  # noqa: E402


@pytest.fixture()
def repo_root(tmp_path: Path) -> Path:
    """Create a minimal repo-root layout inside tmp_path."""
    (tmp_path / "registry").mkdir()
    # Start with an empty forkies registry
    (tmp_path / "registry" / "forkies.yaml").write_text("[]\n")
    # fork_model uses relative paths, so we need to chdir
    import os
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(old_cwd)


class TestForkModel:
    def test_creates_directory(self, repo_root):
        fork_model("Qwen/Qwen2.5-Coder-7B-Instruct", "v1.0.0")
        assert (repo_root / "forkies" / "qwen2-5-coder-7b-instruct" / "v1.0.0").is_dir()

    def test_creates_fork_yaml(self, repo_root):
        fork_model("meta-llama/Llama-3.1-8B-Instruct", "v1.0.0")
        fork_yaml = repo_root / "forkies" / "llama-3-1-8b-instruct" / "v1.0.0" / "FORK.yaml"
        assert fork_yaml.exists()

        with open(fork_yaml) as f:
            meta = yaml.safe_load(f)

        assert meta["source"] == "meta-llama/Llama-3.1-8B-Instruct"
        assert meta["version"] == "v1.0.0"
        assert meta["stage"] == "forkie"
        assert meta["serving_allowed"] is False

    def test_creates_readme(self, repo_root):
        fork_model("Qwen/Qwen2.5-Coder-7B-Instruct", "v1.0.0")
        readme = repo_root / "forkies" / "qwen2-5-coder-7b-instruct" / "v1.0.0" / "README.md"
        assert readme.exists()
        content = readme.read_text()
        assert "Forkie" in content
        assert "Qwen/Qwen2.5-Coder-7B-Instruct" in content

    def test_registers_in_forkies_yaml(self, repo_root):
        fork_model("mistralai/Mistral-7B-Instruct-v0.3", "v0.3.0")
        with open(repo_root / "registry" / "forkies.yaml") as f:
            registry = yaml.safe_load(f)

        ids = [m["id"] for m in registry]
        assert "forkies/mistral-7b-instruct-v0-3@v0.3.0" in ids

    def test_duplicate_fork_is_skipped(self, repo_root):
        fork_model("Qwen/Qwen2.5-Coder-7B-Instruct", "v1.0.0")
        fork_model("Qwen/Qwen2.5-Coder-7B-Instruct", "v1.0.0")

        with open(repo_root / "registry" / "forkies.yaml") as f:
            registry = yaml.safe_load(f)

        matching = [m for m in registry if m["id"] == "forkies/qwen2-5-coder-7b-instruct@v1.0.0"]
        assert len(matching) == 1

    def test_normalizes_model_name(self, repo_root):
        result = fork_model("Qwen/Qwen2.5-Coder-7B-Instruct", "v1.0.0")
        assert result["name"] == "qwen2-5-coder-7b-instruct"

    def test_local_path_used_as_artifact_path(self, repo_root):
        result = fork_model("Qwen/Qwen2.5-Coder-7B-Instruct", "v1.0.0", local_path="/data/weights")
        assert result["artifacts"]["weights"] == "/data/weights"

    def test_s3_path_when_no_local_path(self, repo_root):
        result = fork_model("Qwen/Qwen2.5-Coder-7B-Instruct", "v1.0.0")
        assert result["artifacts"]["weights"].startswith("s3://")
