"""models.toml access + shared paths/DSN. The registry is the single source of
truth for model config; kernel/build.rs reads the same file."""
import os
import tomllib

def repo_root():
    return os.environ.get(
        "PGT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def registry_path():
    return os.environ.get("PGT_REGISTRY", os.path.join(repo_root(), "models.toml"))

def artifacts_dir():
    d = os.environ.get("PGT_ARTIFACTS", os.path.join(repo_root(), "artifacts"))
    os.makedirs(d, exist_ok=True)
    return d

def sql_path():
    return os.path.join(repo_root(), "sql", "pg_transformers.sql")

def load_registry():
    with open(registry_path(), "rb") as f:
        raw = tomllib.load(f)
    # resolve `base = "<key>"` inheritance (variant entries, e.g. int8)
    reg = {}
    for k, m in raw.items():
        if "base" in m:
            if m["base"] not in raw:
                raise KeyError(f"{k}.base = '{m['base']}' not in {registry_path()}")
            reg[k] = {**raw[m["base"]], **m}
        else:
            reg[k] = dict(m)
    return reg

def model(key):
    reg = load_registry()
    if key not in reg:
        raise KeyError(f"model '{key}' not in {registry_path()} (have: {', '.join(reg)})")
    return reg[key]

def default_dsn():
    return os.environ.get("PGT_DSN", "host=localhost port=5432 user=postgres dbname=postgres")
