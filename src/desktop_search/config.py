from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class Settings(BaseModel):
    folders: list[Path] = Field(default_factory=list)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 6
    confidence_threshold: float = 0.45
    chroma_path: Path = Path("./data/chroma")
    llm_model: str = "claude-sonnet-4-6"
    ignore_patterns: list[str] = Field(
        default_factory=lambda: [".git", ".venv", "venv", "node_modules", "__pycache__", ".DS_Store"]
    )


def load_settings(path: Path = Path("config.yaml")) -> Settings:
    if not path.exists():
        return Settings()
    data = yaml.safe_load(path.read_text()) or {}
    return Settings.model_validate(data)
