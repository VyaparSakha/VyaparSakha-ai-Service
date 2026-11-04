from functools import lru_cache
from dotenv import load_dotenv

from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    gemini_api_key: str | None = None
    mongo_url: str = "mongodb://localhost:27017"
    mongo_db_name: str = "vyapar-sathi"
    redis_url: str | None = None
    langchain_api_key: str | None = None
    langchain_project: str = "vyaparSathi-ai"
    langchain_verbose: bool = True
    langchain_tracing_v2: bool = True
    langgraph_checkpointer_ttl_seconds: int = 604800
    # Mem0 long-term memory
    mem0_api_key: str | None = None
    mem0_enabled: bool = True
    # Small/fast summarizer providers (OpenAI-compatible endpoints)
    groq_api_key: str | None = None
    nvidia_api_key: str | None = None
    openrouter_api_key: str | None = None
    ollama_base_url: str | None = None
    # Vector database (Pinecone)
    pinecone_api_key: str | None = None
    pincone_api_key: str | None = None
    pinecone_index_name: str = "vyapar-sathi"
    # Web search
    tavily_api_key: str | None = None
    # Web search / scraping (Firecrawl)
    firecrawl_api_key: str | None = None
    firecrawl_api_url: str = "https://api.firecrawl.dev"
    firecrawl_timeout_seconds: float = 60.0
    # Cloudflare R2 object storage
    cloudflare_account_id: str | None = None
    cloudflare_r2_endpoint: str | None = None
    cloudflare_r2_access_key_id: str | None = None
    cloudflare_r2_secret_access_key: str | None = None
    cloudflare_r2_bucket_name: str | None = None

    # Express Backend URL for communications
    express_backend_url: str = "http://localhost:5000"
    internal_service_key: str = "vyapar-internal-ai-service-key"

    # Document generation limits
    max_generated_file_size_mb: int = 10


    @property
    def effective_pinecone_api_key(self) -> str | None:
        return self.pinecone_api_key or self.pincone_api_key

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
