import pathlib
import sys

import uvicorn

# Ensure project root is on sys.path when running `python server/main.py`
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=7788, reload=False)


if __name__ == "__main__":
    main()
