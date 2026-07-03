import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "services" / "inference-service"))

from app.predictor import build_segmentation_feature_frame, build_support_feature_frame  # noqa: E402
from app.schemas import SegmentationFeatures, SupportForecastFeatures  # noqa: E402


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


def test_segmentation_feature_frame_contains_expected_features() -> None:
    features = SegmentationFeatures(
        server_id="S000000",
        server_type="vps",
        region="waw",
        os_family="linux",
        segment="startup",
        country="DE",
        support_plan="critical",
        cpu_cores=8,
        ram_gb=16,
        disk_tb=2.0,
        age_days=1339,
        has_gpu=0,
        is_managed=0,
        cpu_util_pct=42.27,
        ram_util_pct=41.65,
        disk_util_pct=35.07,
        net_in_gb=248.26,
        net_out_gb=297.19,
        temperature_c=52.19,
        backup_success=1,
        scheduled_maintenance=0,
        avg_rack_temperature_c=54.13,
        power_usage_mw=0.625,
        network_latency_ms=18.55,
        capacity_used_pct=68.56,
        contract_months=36,
        tenure_days=1197,
        monthly_spend_eur=130.34,
        observation_count=34,
    )

    frame = build_segmentation_feature_frame(features)

    assert frame.shape == (1, 51)
    assert frame.loc[0, "net_in_gb_sum"] == 248.26 * 34
    assert frame.loc[0, "cpu_util_pct_std"] == 0
    assert frame.loc[0, "backup_failure_rate"] == 0
