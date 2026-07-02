import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "services" / "inference-service"))

from app.predictor import build_support_feature_frame  # noqa: E402
from app.schemas import SupportForecastFeatures  # noqa: E402


def test_support_feature_frame_contains_expected_features() -> None:
    features = SupportForecastFeatures(
        date="2026-03-17",
        region="waw",
        scheduled_maintenance=0,
        avg_rack_temperature_c=53.59,
        power_usage_mw=0.65,
        network_latency_ms=23.24,
        capacity_used_pct=66.77,
        recent_support_tickets=5,
    )

    frame = build_support_feature_frame(features)

    assert frame.shape == (1, 47)
    assert frame.loc[0, "day_of_week"] == 1
    assert frame.loc[0, "support_tickets_lag1"] == 5
    assert frame.loc[0, "support_tickets_rolling_std_7"] == 0
