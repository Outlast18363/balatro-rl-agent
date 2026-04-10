from pathlib import Path

from run_async_rollout import build_demo_specs


def test_build_demo_specs_plain_seeds():
    specs = build_demo_specs(3, seed=7)

    assert len(specs) == 3
    assert [spec.seed for spec in specs] == [7, 8, 9]
    assert all(spec.snapshot is None for spec in specs)
    assert all(spec.save_source is None for spec in specs)


def test_build_demo_specs_attaches_save_source_to_last_worker():
    save_path = Path("game_files/first_blind_combat_save.jkr")

    specs = build_demo_specs(3, seed=10, save_source=save_path)

    assert specs[-1].save_source == save_path
    assert specs[-1].seed == 12
