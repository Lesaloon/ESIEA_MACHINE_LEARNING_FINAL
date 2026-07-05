from pathlib import Path


def test_expected_directories_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "data").exists()
    assert (root / "models").exists()
    for service_name in [
        "frontend",
        "api-gateway",
        "incident-inference-service",
        "support-inference-service",
        "segmentation-inference-service",
        "anomaly-inference-service",
        "shared-inference",
    ]:
        assert (root / "services" / service_name).exists()
