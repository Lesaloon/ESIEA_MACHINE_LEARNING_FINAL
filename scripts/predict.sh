#!/usr/bin/env sh
set -eu

curl -sS -X POST "http://localhost:${INFERENCE_SERVICE_PORT:-8000}/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {
      "date": "2026-03-17",
      "server_id": "S000000",
      "server_type": "vps",
      "region": "waw",
      "os_family": "linux",
      "cpu_cores": 8,
      "ram_gb": 16,
      "disk_tb": 2.0,
      "age_days": 1339,
      "has_gpu": 0,
      "is_managed": 0,
      "cpu_util_pct": 82.84,
      "ram_util_pct": 46.65,
      "disk_util_pct": 28.8,
      "net_in_gb": 249.31,
      "net_out_gb": 191.95,
      "temperature_c": 63.74,
      "backup_success": 1,
      "scheduled_maintenance": 0,
      "avg_rack_temperature_c": 53.59,
      "power_usage_mw": 0.65,
      "network_latency_ms": 23.24,
      "support_tickets": 5,
      "capacity_used_pct": 66.77,
      "segment": "startup",
      "country": "DE",
      "contract_months": 36,
      "support_plan": "critical",
      "tenure_days": 1197,
      "monthly_spend_eur": 130.34
    }
  }'
