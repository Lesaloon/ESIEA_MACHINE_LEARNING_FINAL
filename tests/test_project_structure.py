from pathlib import Path


def test_expected_directories_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "data").exists()
    assert (root / "models").exists()
    assert (root / "services" / "inference-service").exists()
