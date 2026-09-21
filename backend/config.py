from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_key: str = os.getenv("SUPABASE_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    backend_host: str = os.getenv("BACKEND_HOST", "0.0.0.0")
    backend_port: int = int(os.getenv("BACKEND_PORT", "8000"))
    frontend_api_base_url: str = os.getenv("FRONTEND_API_BASE_URL", "http://localhost:8000")
    hybrid_match_count: int = int(os.getenv("HYBRID_MATCH_COUNT", "8"))
    final_context_chunks: int = int(os.getenv("FINAL_CONTEXT_CHUNKS", "4"))
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))

    def validate_required(self) -> None:
        missing = []
        if not self.supabase_url:
            missing.append("SUPABASE_URL")
        if not self.supabase_key:
            missing.append("SUPABASE_KEY")
        if not self.groq_api_key:
            missing.append("GROQ_API_KEY")
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(f"Missing required environment variables: {joined}")


settings = Settings()
