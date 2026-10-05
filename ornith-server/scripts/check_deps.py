import importlib

for name in ["fastapi", "uvicorn", "httpx", "yaml", "pydantic", "openai", "starlette"]:
    try:
        mod = importlib.import_module(name)
        print(f"{name:12} OK      {getattr(mod, '__version__', '?')}")
    except Exception as exc:
        print(f"{name:12} MISSING {exc}")
