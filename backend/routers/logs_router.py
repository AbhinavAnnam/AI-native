import os
from fastapi import APIRouter
from deps import LOG_FILE

router = APIRouter(prefix="/api", tags=["Logs"])


@router.get("/logs")
async def get_logs():
    if not os.path.exists(LOG_FILE):
        return {"logs": "No log file found."}
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            logs = f.read()
        return {"logs": logs if logs.strip() else "Log file is empty."}
    except Exception as exc:
        return {"logs": f"Error reading log file: {str(exc)}"}