"""
Unit tests for tools/promote.py - check_promotion_criteria and promote_model.
"""
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from promote import check_promotion_criteria, promote_model  # noqa: E402


def _write_manifest(path: Path, stage: str, evals=None, has_lineage: bool = True) -> None:
    path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "id": f"{stage}/test-model-v1",
        "name": "test-model",
        "stage": stage,
        "version": "v1",
        "evals": evals or [],
    }
    with open(path / "MANIFEST.yaml", "w") as f:
        yaml.dump(manifest, f)
    if has_lineage:
        (path / "LINEAGE.md").write_text("# Lineage\nDerived from forkie.\n")


@pytest.fixture()
def repo_root(tmp_path: Path):
    """Set up a minimal repo root for promotion tests."""
    import os

    (tmp_path / "registry").mkdir()
    for stage in ("internal", "production"):
        (tmp_path / "registry" / f"{stage}.yaml").write_text("[]\n")
    (tmp_path / "registry" / "lineage.yaml").write_text("[]\n")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(old_cwd)


class TestCheckPromotionCriteria:
    def test_valid_research_to_internal_passes(self, repo_root):
        model_dir = repo_root / "research" / "alexa" / "my-model"
        _write_manifest(
            model_dir,
            "research",
            evals=[{"name": "HumanEval", "score": 0.72}],
        )
        assert check_promotion_criteria(model_dir, "internal", auto_yes=True) is True

    def test_research_humaneval_below_threshold_fails(self, repo_root):
        model_dir = repo_root / "research" / "alexa" / "weak-model"
        _write_manifest(
            model_dir,
            "research",
            evals=[{"name": "HumanEval", "score": 0.55}],
        )
        assert check_promotion_criteria(model_dir, "internal", auto_yes=True) is False

    def test_invalid_promotion_path_fails(self, repo_root):
        model_dir = repo_root / "research" / "alexa" / "my-model"
        _write_manifest(model_dir, "research")
        assert check_promotion_criteria(model_dir, "production", auto_yes=True) is False

    def test_forkie_cannot_be_promoted(self, repo_root):
        model_dir = repo_root / "forkies" / "some-model" / "v1.0.0"
        _write_manifest(model_dir, "forkie")
        assert check_promotion_criteria(model_dir, "internal", auto_yes=True) is False

    def test_internal_to_production_requires_auto_yes(self, repo_root):
        model_dir = repo_root / "internal" / "my-model" / "v1"
        _write_manifest(model_dir, "internal", evals=[{"name": "HumanEval", "score": 0.75}])
        assert check_promotion_criteria(model_dir, "production", auto_yes=True) is True

    def test_no_manifest_raises(self, repo_root):
        model_dir = repo_root / "research" / "alexa" / "no-manifest"
        model_dir.mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            check_promotion_criteria(model_dir, "internal", auto_yes=True)


class TestPromoteModel:
    def test_research_to_internal_succeeds(self, repo_root):
        model_dir = repo_root / "research" / "alexa" / "my-model"
        _write_manifest(
            model_dir,
            "research",
            evals=[{"name": "HumanEval", "score": 0.72}],
        )

        result = promote_model(str(model_dir), "internal", new_name="blackroad-coder-7b", auto_yes=True)
        assert result is True

        target = repo_root / "internal" / "blackroad-coder-7b" / "v1"
        assert target.is_dir()
        assert (target / "MANIFEST.yaml").exists()

    def test_promoted_manifest_has_correct_stage(self, repo_root):
        model_dir = repo_root / "research" / "alexa" / "my-model"
        _write_manifest(
            model_dir,
            "research",
            evals=[{"name": "HumanEval", "score": 0.72}],
        )
        promote_model(str(model_dir), "internal", new_name="blackroad-coder-7b", auto_yes=True)

        target_manifest = repo_root / "internal" / "blackroad-coder-7b" / "v1" / "MANIFEST.yaml"
        with open(target_manifest) as f:
            manifest = yaml.safe_load(f)
        assert manifest["stage"] == "internal"
        assert manifest["promoted_from"] is not None

    def test_promote_adds_to_registry(self, repo_root):
        model_dir = repo_root / "research" / "alexa" / "my-model"
        _write_manifest(
            model_dir,
            "research",
            evals=[{"name": "HumanEval", "score": 0.72}],
        )
        promote_model(str(model_dir), "internal", new_name="blackroad-coder-7b", auto_yes=True)

        with open(repo_root / "registry" / "internal.yaml") as f:
            registry = yaml.safe_load(f)
        assert registry is not None
        ids = [m["id"] for m in registry]
        assert any("blackroad-coder-7b" in rid for rid in ids)

    def test_promote_updates_lineage(self, repo_root):
        model_dir = repo_root / "research" / "alexa" / "my-model"
        _write_manifest(
            model_dir,
            "research",
            evals=[{"name": "HumanEval", "score": 0.72}],
        )
        promote_model(str(model_dir), "internal", new_name="blackroad-coder-7b", auto_yes=True)

        with open(repo_root / "registry" / "lineage.yaml") as f:
            lineage = yaml.safe_load(f)
        assert lineage is not None
        assert len(lineage) >= 1

    def test_source_not_found_raises(self, repo_root):
        with pytest.raises(FileNotFoundError):
            promote_model("nonexistent/path", "internal", new_name="x", auto_yes=True)

    def test_missing_name_raises(self, repo_root):
        model_dir = repo_root / "research" / "alexa" / "my-model"
        _write_manifest(
            model_dir,
            "research",
            evals=[{"name": "HumanEval", "score": 0.72}],
        )
        with pytest.raises(ValueError, match="--name"):
            promote_model(str(model_dir), "internal", new_name=None, auto_yes=True)
