import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import uvicorn
from fastapi import FastAPI
from app.api.talents import router

app = FastAPI(title="Talent Reply Agent", version="1.0.0")
app.include_router(router, prefix="/api/talents")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
