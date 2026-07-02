from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    inputs: dict[str, Any] | None = None


class IncidentFeatures(BaseModel):
    date: str = "2026-03-17"
    server_id: str = "SAMPLE_SERVER"
    server_type: str
    region: str
    os_family: str
    cpu_cores: int
    ram_gb: int
    disk_tb: float
    age_days: int
    has_gpu: int
    is_managed: int
    cpu_util_pct: float
    ram_util_pct: float
    disk_util_pct: float
    net_in_gb: float
    net_out_gb: float
    temperature_c: float
    backup_success: int
    scheduled_maintenance: int
    avg_rack_temperature_c: float
    power_usage_mw: float
    network_latency_ms: float
    support_tickets: int
    capacity_used_pct: float
    segment: str
    country: str
    contract_months: int
    support_plan: str
    tenure_days: int
    monthly_spend_eur: float
    day_of_week: int | None = None
    day_of_month: int | None = None
    days_since_start: int | None = None


class PredictionResponse(BaseModel):
    prediction: int
    incident_probability: float
    risk_level: str
    metadata: dict[str, Any] = Field(default_factory=dict)
