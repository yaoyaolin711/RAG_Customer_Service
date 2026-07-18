import os
from pathlib import Path
from typing import Optional
import yaml
from dotenv import load_dotenv


class Config:
    _instance: Optional["Config"] = None
    _config: dict = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _find_config_dir(self) -> Path:
        candidates = [
            Path(__file__).resolve().parent.parent,
            Path(__file__).resolve().parent,
        ]
        env_dir = os.getenv("CRM_AGENT_CONFIG_DIR")
        if env_dir:
            candidates.insert(0, Path(env_dir))
        for d in candidates:
            if (d / "config.yaml").exists():
                return d
        return candidates[0]

    def _load_config(self):
        cfg_dir = self._find_config_dir()

        load_dotenv(cfg_dir / ".env")

        config_path = cfg_dir / "config.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}

        agents_config_path = cfg_dir / "agents.yaml"
        if agents_config_path.exists():
            with open(agents_config_path, "r", encoding="utf-8") as f:
                agents_config = yaml.safe_load(f)
                if agents_config:
                    self._config["agents"] = agents_config

        self._resolve_env_vars()

    def _resolve_env_vars(self):
        def resolve_value(val):
            if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
                env_key = val[2:-1]
                return os.getenv(env_key, "")
            return val

        def resolve_dict(d):
            if isinstance(d, dict):
                return {k: resolve_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [resolve_dict(item) for item in d]
            else:
                return resolve_value(d)

        self._config = resolve_dict(self._config)

    @property
    def llm(self) -> dict:
        return self._config.get("llm", {})

    @property
    def agents(self) -> dict:
        return self._config.get("agents", {})

    @property
    def streaming(self) -> dict:
        return self._config.get("streaming", {})

    @property
    def context(self) -> dict:
        return self._config.get("context", {})

    def get(self, key: str, default=None):
        return self._config.get(key, default)


config = Config()
