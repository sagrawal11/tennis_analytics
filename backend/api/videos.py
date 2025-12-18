from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
import os
from supabase import create_client, Client
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from auth import get_user_id

router = APIRouter(prefix="/api/videos", tags=["videos"])

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not supabase_url or not supabase_key:
    raise ValueError("Supabase credentials not configured")

supabase: Client = create_client(supabase_url, supabase_key)


class VideoUpload(BaseModel):
    match_id: str
    playsight_link: str


class PlayerIdentification(BaseModel):
    match_id: str
    frame_data: dict
    selected_player_coords: dict


@router.post("/upload")
async def upload_video(video_data: VideoUpload, user_id: str = Depends(get_user_id)):
    """
    Submit a video for processing.
    Updates the match with the playsight link and sets status to pending.
    """
    # Verify match belongs to user
    match_response = supabase.table("matches").select("*").eq("id", video_data.match_id).single().execute()
    
    if not match_response.data:
        raise HTTPException(status_code=404, detail="Match not found")
    
    match = match_response.data
    if match["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Update match
    update_response = supabase.table("matches").update({
        "playsight_link": video_data.playsight_link,
        "status": "pending"
    }).eq("id", video_data.match_id).execute()
    
    return {"message": "Video submitted for processing", "match": update_response.data[0] if update_response.data else match}


@router.post("/identify-player")
async def identify_player(identification: PlayerIdentification, user_id: str = Depends(get_user_id)):
    """
    Store player identification data.
    This will be used by the CV backend to track the specific player.
    """
    # Verify match belongs to user
    match_response = supabase.table("matches").select("id").eq("id", identification.match_id).eq("user_id", user_id).execute()
    
    if not match_response.data:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Store identification
    ident_response = supabase.table("player_identifications").insert({
        "match_id": identification.match_id,
        "frame_data": identification.frame_data,
        "selected_player_coords": identification.selected_player_coords
    }).execute()
    
    if not ident_response.data:
        raise HTTPException(status_code=500, detail="Failed to store identification")
    
    # Update match status to processing
    supabase.table("matches").update({
        "status": "processing"
    }).eq("id", identification.match_id).execute()
    
    # Trigger CV processing (async - would use task queue in production)
    try:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        from services.cv_integration import trigger_processing
        
        # In production, this would be done via Celery or similar
        # For now, we'll just mark it - actual processing would happen async
        # trigger_processing(identification.match_id)
    except Exception as e:
        # Log error but don't fail the request
        print(f"Error triggering processing: {e}")
    
    return {"message": "Player identification stored", "identification": ident_response.data[0]}


@router.get("/{match_id}/status")
async def get_processing_status(match_id: str, user_id: str = Depends(get_user_id)):
    """Get the processing status of a match."""
    match_response = supabase.table("matches").select("status, processed_at").eq("id", match_id).single().execute()
    
    if not match_response.data:
        raise HTTPException(status_code=404, detail="Match not found")
    
    # Verify access
    match = supabase.table("matches").select("user_id").eq("id", match_id).single().execute()
    if match.data and match.data["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return {
        "status": match_response.data["status"],
        "processed_at": match_response.data.get("processed_at")
    }
