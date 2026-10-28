"""Configuration loading.

Business parameters come from ``config.yaml``; secrets come from ``.env``.
Precedence (highest first): CLI overrides > environment variables > YAML > defaults.
``NASDAQ_ENV_FILE`` points at another env file, or ``none`` to read only the process
environment (the test suite uses this so it never sees a developer's credentials).
Personal addresses are excluded from ``repr`` so tracebacks and test failures cannot print them.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class AnalysisConfig(BaseModel):
    days: int = Field(5, ge=2, le=30)
    benchmarks: dict[str, str] = {"^GSPC": "SPY", "^IXIC": "QQQ"}


class ScreenerFilters(BaseModel):
    exclude_derivatives: bool = True
    min_price: float = Field(1, ge=0)
    min_volume: int = Field(100_000, ge=0)
    min_market_cap: float = Field(0, ge=0)


class ScreenerConfig(BaseModel):
    top_n: int = Field(10, ge=1, le=50)
    default_filters: ScreenerFilters = ScreenerFilters()


class DataChecksConfig(BaseModel):
    cross_source_warn_pp: float = 1.0
    cross_source_conflict_pp: float = 5.0
    outlier_daily_move: float = 0.8
    news_max_age_days: int = 7
    filter_boundary_band: float = Field(0.10, ge=0, le=0.5)   # engineering default, not yet calibrated
    max_verification_requests: int = Field(12, ge=0, le=50)
    verification_budget_s: float = Field(25, gt=0)


class ProviderCredentials(BaseModel):
    name: str
    api_key: SecretStr
    base_url: str
    model: str
    output_token_parameter: Literal["max_tokens", "max_completion_tokens"] | None = None


class LLMConfig(BaseModel):
    provider: str = "mimo"
    fallback: list[str] = ["deepseek"]
    temperature: float = 0.0
    timeout_s: float = Field(60, gt=0, allow_inf_nan=False)            # fallback providers
    primary_timeout_s: float = Field(25, gt=0, allow_inf_nan=False)    # primary: fail over fast
    max_retries: int = Field(2, ge=0, le=10)
    max_output_tokens: int = Field(8192, ge=1)


class JevConfig(BaseModel):
    enabled: bool = True
    allow_rule_fallback: bool = False   # reviewed fixtures only; rules cannot establish factual support
    min_confidence: float = 0.5
    consistency_threshold: float = 0.3
    unsupported_fact_threshold: float = Field(0.5, ge=0, le=1)
    speculation_threshold: float = 0.5  # calibrated with scripts/eval_speculation.py
    api_key: SecretStr | None = None
    base_url: str = "https://api.typesafe.ai"
    model: str = "jev-latest"


class SandboxConfig(BaseModel):
    backend: Literal["docker", "trusted_local"] = "docker"
    image: str = "nasdaq-analysis:local"
    memory_mb: int = Field(512, ge=128, le=4096)
    cpus: float = Field(1, gt=0, le=8)
    timeout_s: float = 30
    max_attempts: int = 3
    max_stdout_chars: int = 10_000


class RunConfig(BaseModel):
    deadline_s: float = 420
    max_llm_calls: int = 30
    max_guard_rejections: int = 2
    recursion_limit: int = 60
    breaker_failures: int = 3
    output_dir: Path = Path("outputs")


class ModelPrice(BaseModel):
    version: str
    input_per_million: float = Field(ge=0, allow_inf_nan=False)
    output_per_million: float = Field(ge=0, allow_inf_nan=False)
    cached_input_per_million: float | None = Field(None, ge=0, allow_inf_nan=False)
    per_request: float = Field(0, ge=0, allow_inf_nan=False)


class BudgetConfig(BaseModel):
    # Extra limits are opt-in. max_llm_calls still limits agent turns.
    max_model_requests: int | None = Field(None, ge=1)
    max_total_tokens: int | None = Field(None, ge=1)
    max_estimated_cost_usd: float | None = Field(None, gt=0, allow_inf_nan=False)
    # Explicit provider:model keys; no assumed prices or free unknown usage.
    prices: dict[str, ModelPrice] = Field(default_factory=dict)


class SMTPConfig(BaseModel):
    host: str = "smtp.gmail.com"
    port: int = 587
    user: str | None = Field(None, repr=False)
    password: SecretStr | None = None


class EmailConfig(BaseModel):
    to: list[str] = Field(default_factory=list, repr=False)
    sender: str | None = Field(None, repr=False)
    dry_run: bool = True
    include_chart: bool = True
    narrative_max_words: int = Field(250, ge=1)  # upper bound only; no minimum length
    smtp: SMTPConfig = SMTPConfig()


class Settings(BaseModel):
    analysis: AnalysisConfig = AnalysisConfig()
    screener: ScreenerConfig = ScreenerConfig()
    data_checks: DataChecksConfig = DataChecksConfig()
    llm: LLMConfig = LLMConfig()
    providers: dict[str, ProviderCredentials] = {}
    jev: JevConfig = JevConfig()
    sandbox: SandboxConfig = SandboxConfig()
    run: RunConfig = RunConfig()
    budget: BudgetConfig = BudgetConfig()
    email: EmailConfig = EmailConfig()
    api_token: SecretStr | None = None

    @property
    def output_root(self) -> Path:
        root = self.run.output_dir
        return root if root.is_absolute() else PROJECT_ROOT / root


def _env(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


def _env_bool(name: str) -> bool | None:
    value = _env(name)
    return None if value is None else value.lower() in {"1", "true", "yes", "on"}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict):
            # Recurse even when the base lacks the key, so unset (None) leaves keep their defaults.
            merged[key] = _deep_merge(merged.get(key) if isinstance(merged.get(key), dict) else {}, value)
        elif value is not None:
            merged[key] = value
    return merged


def _env_layer() -> dict[str, Any]:
    """Map environment variables onto the settings tree."""
    providers = {}
    for name in ("mimo", "deepseek", "openai"):
        prefix = name.upper()
        key, url, model = _env(f"{prefix}_API_KEY"), _env(f"{prefix}_BASE_URL"), _env(f"{prefix}_MODEL")
        if key and url and model:
            providers[name] = {"name": name, "api_key": key, "base_url": url, "model": model}

    fallback = _env("LLM_FALLBACK_ORDER")
    provider = _env("LLM_PROVIDER")
    # Keep the full order (duplicates of the primary are dropped later), so switching the primary
    # with --provider leaves the other providers as fallbacks.
    fallback_list = [p.strip() for p in fallback.split(",") if p.strip()] if fallback else None

    to = _env("EMAIL_TO")
    return {
        "api_token": _env("NASDAQ_API_TOKEN"),
        "run": {"output_dir": _env("RUN_OUTPUT_DIR")},
        "sandbox": {"backend": _env("SANDBOX_BACKEND"), "image": _env("SANDBOX_IMAGE")},
        "llm": {"provider": provider, "fallback": fallback_list},
        "providers": providers,
        "jev": {
            "enabled": _env_bool("JEV_ENABLED"),
            "api_key": _env("TYPESAFE_API_KEY"),
            "base_url": _env("TYPESAFE_BASE_URL"),
            "model": _env("JEV_MODEL"),
        },
        "email": {
            "to": [t.strip() for t in to.split(",")] if to else None,
            "sender": _env("EMAIL_FROM") or _env("SMTP_USER"),
            "dry_run": _env_bool("EMAIL_DRY_RUN"),
            "smtp": {
                "host": _env("SMTP_HOST"),
                "port": int(_env("SMTP_PORT")) if _env("SMTP_PORT") else None,
                "user": _env("SMTP_USER"),
                "password": (_env("SMTP_PASSWORD") or "").replace(" ", "") or None,
            },
        },
    }


def load_settings(
    config_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
    env_file: Path | None = None,
) -> Settings:
    """Build settings from YAML, environment and CLI overrides."""
    chosen = env_file or os.getenv("NASDAQ_ENV_FILE") or PROJECT_ROOT / ".env"
    if str(chosen).strip().lower() != "none":
        load_dotenv(chosen, override=False)
    path = config_path or PROJECT_ROOT / "config.yaml"
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    yaml_to = (data.get("email") or {}).get("to")
    env_layer = _env_layer()
    if yaml_to:  # an explicit list in config.yaml wins over EMAIL_TO
        env_layer["email"]["to"] = None
    data = _deep_merge(data or {}, env_layer)
    data = _deep_merge(data, overrides or {})
    return Settings.model_validate(data)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


ProviderName = Literal["mimo", "deepseek", "openai"]
