from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(
    title="Tennis Analytics API",
    description="Backend API for tennis analytics application",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
from api import teams, matches, videos, stats

app.include_router(teams.router)
app.include_router(matches.router)
app.include_router(videos.router)
app.include_router(stats.router)


@app.get("/")
async def root():
    return {"message": "Tennis Analytics API", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/api/health")
async def api_health():
    return {"status": "ok", "message": "API is running"}
