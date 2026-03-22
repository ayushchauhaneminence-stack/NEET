import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Settings:
    CHANDRA_API_KEY: str = os.getenv("CHANDRA_API_KEY", "")
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    ZILLIZ_URI: str = os.getenv("ZILLIZ_URI", "")
    ZILLIZ_TOKEN: str = os.getenv("ZILLIZ_TOKEN", "")
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-embed-text-v2-moe")
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", str(Path(__file__).parents[2] / "uploads"))


settings = Settings()
