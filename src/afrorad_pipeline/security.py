import os


def get_env_token(var_name: str, required: bool = True) -> str | None:
    token = os.getenv(var_name)
    if token:
        return token.strip()
    if required:
        raise RuntimeError(
            f"Missing required environment variable: {var_name}. "
            "Set it before running the pipeline."
        )
    return None
