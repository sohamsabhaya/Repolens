"""
Global Configuration & Environment Settings
"""

import os
from pathlib import Path
import dotenv

# Load environment variables from .env
dotenv.load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment or .env file!")

# Model Configurations
PRIMARY_LLM_MODEL = os.getenv("PRIMARY_LLM_MODEL", "qwen/qwen3.8-27b")
FAST_LLM_MODEL = os.getenv("FAST_LLM_MODEL", "qwen/qwen3.8-27b")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_EMBEDDING_MODEL = EMBEDDING_MODEL_NAME

# Directories & Caching
BASE_DIR = Path(__file__).resolve().parent.parent
INDEXES_DIR = BASE_DIR / ".indexes"
CLONED_REPOS_DIR = BASE_DIR / "cloned_repos"

# Parsing Constants
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_MAX_WORKERS = min(32, (os.cpu_count() or 4) * 4)

IGNORE_DIRS = {
    ".git", "venv", ".venv", "__pycache__", "node_modules", 
    ".idea", ".vscode", "dist", "build", "target", "bin", 
    ".gradle", ".mvn", "out"
}

IGNORE_EXTENSIONS = {
    ".pyc", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", 
    ".zip", ".tar", ".gz", ".exe", ".dll", ".so", ".dylib", 
    ".class", ".jar", ".war", ".ear"
}

VALID_CONFIG_EXTENSIONS = {
    ".txt", ".toml", ".yaml", ".yml", ".json", ".sql", 
    ".sh", ".env.example", ".xml", ".properties", ".gradle"
}
