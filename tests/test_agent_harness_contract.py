from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_agents_md_describes_control_plane_split():
    agents_md = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")

    assert "scripts/control_plane.py" in agents_md
    assert "parallel_train.py" in agents_md
    assert "parallel_eval.py" in agents_md
    assert "conda activate regmixer" in agents_md
    assert "帮我跑这个实验" in agents_md


def test_skill_mentions_control_plane_launch_loop():
    skill_md = (REPO_ROOT / ".agents" / "skills" / "parallel-train-operator" / "SKILL.md").read_text(
        encoding="utf-8"
    )

    assert "scripts/control_plane.py" in skill_md
    assert "launch loop" in skill_md.lower()
    assert "parallel_train.py" in skill_md
    assert "parallel_eval.py" in skill_md


def test_monitoring_doc_references_control_and_executor_state():
    monitoring_md = (REPO_ROOT / ".agents" / "docs" / "hpcgpu-monitoring-loop.md").read_text(encoding="utf-8")

    assert "control_plane_state.json" in monitoring_md
    assert "parallel_train_state.json" in monitoring_md
    assert "parallel_eval_state.json" in monitoring_md
