from pathlib import Path

token_path = Path(__file__).parent.parent.parent / "model" / "llama-3.1-8B-instruct" / "token.txt"

def load_token() -> str:
    with open(token_path, "r") as f:
        token = f.read()
    return token
