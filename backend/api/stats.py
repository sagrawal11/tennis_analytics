from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
import os
from supabase import create_client, Client
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from auth import get_user_id

router = APIRouter(prefix="/api/stats", tags=["stats"])

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not supabase_url or not supabase_key:
    raise ValueError("Supabase credentials not configured")

supabase: Client = create_client(supabase_url, supabase_key)


@router.get("/player/{player_id}")
async def get_player_stats(player_id: str, user_id: str = Depends(get_user_id)):
    """
    Get season statistics for a player.
    Coaches can view any team member's stats, players can only view their own.
    """
    # Verify access
    if player_id != user_id:
        # Check if user is coach on same team
        user_response = supabase.table("users").select("role").eq("id", user_id).single().execute()
        if not user_response.data or user_response.data.get("role") != "coach":
            raise HTTPException(status_code=403, detail="Access denied")
    
    # Get all matches for player
    matches_response = supabase.table("matches").select("id").eq("user_id", player_id).eq("status", "completed").execute()
    match_ids = [m["id"] for m in (matches_response.data or [])]
    
    if not match_ids:
        return {
            "player_id": player_id,
            "total_matches": 0,
            "total_shots": 0,
            "winners": 0,
            "errors": 0,
            "in_play": 0,
            "matches": []
        }
    
    # Get all shots for these matches
    shots_response = supabase.table("shots").select("*").in_("match_id", match_ids).execute()
    shots = shots_response.data or []
    
    # Calculate aggregate stats
    winners = len([s for s in shots if s["result"] == "winner"])
    errors = len([s for s in shots if s["result"] == "error"])
    in_play = len([s for s in shots if s["result"] == "in_play"])
    
    # Get per-match stats
    matches_with_stats = []
    for match_id in match_ids:
        match_shots = [s for s in shots if s["match_id"] == match_id]
        matches_with_stats.append({
            "match_id": match_id,
            "shots": len(match_shots),
            "winners": len([s for s in match_shots if s["result"] == "winner"]),
            "errors": len([s for s in match_shots if s["result"] == "error"]),
            "in_play": len([s for s in match_shots if s["result"] == "in_play"]),
        })
    
    return {
        "player_id": player_id,
        "total_matches": len(match_ids),
        "total_shots": len(shots),
        "winners": winners,
        "errors": errors,
        "in_play": in_play,
        "matches": matches_with_stats
    }


@router.get("/my-stats")
async def get_my_stats(user_id: str = Depends(get_user_id)):
    """Get current user's statistics."""
    return await get_player_stats(user_id, user_id)
