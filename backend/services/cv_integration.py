"""
CV Backend Integration Service.

This module integrates with the existing tennis_analytics CV backend
to process videos and extract shot data.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from supabase import create_client, Client

# Initialize Supabase
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if supabase_url and supabase_key:
    supabase: Client = create_client(supabase_url, supabase_key)
else:
    supabase = None


def process_video(match_id: str, video_path: str, player_coords: Dict) -> Dict:
    """
    Process a video using the CV backend.
    
    Args:
        match_id: Match ID in database
        video_path: Path to video file
        player_coords: Player identification coordinates
        
    Returns:
        Dictionary with processing results
    """
    # Path to CV backend (assuming it's in the old/ directory)
    project_root = Path(__file__).parent.parent.parent
    cv_backend_path = project_root / "old" / "src" / "core" / "tennis_CV.py"
    
    # Check if CV backend exists
    if not cv_backend_path.exists():
        raise FileNotFoundError(f"CV backend not found at {cv_backend_path}")
    
    # TODO: Call CV backend with appropriate parameters
    # This would involve:
    # 1. Running the CV processing script
    # 2. Passing video path and player coordinates
    # 3. Capturing JSON output
    # 4. Parsing and storing results
    
    # Placeholder implementation
    output_data = {
        "shots": [],
        "stats": {},
    }
    
    return output_data


def parse_cv_output(output_json: Dict) -> Dict:
    """
    Parse JSON output from CV backend into structured format.
    
    Expected format:
    {
        "shots": [
            {
                "start_pos": {"x": 10, "y": 20},
                "end_pos": {"x": 80, "y": 90},
                "timestamp": 1234,
                "video_timestamp": 45.6,
                "result": "winner" | "error" | "in_play",
                "shot_type": "forehand" | "backhand" | "serve" | etc.
            }
        ],
        "stats": {
            "total_shots": 150,
            "winners": 25,
            "errors": 30,
            "first_serve_percentage": 65.5,
            ...
        }
    }
    """
    shots = output_json.get("shots", [])
    stats = output_json.get("stats", {})
    
    return {
        "shots": shots,
        "stats": stats,
    }


def store_match_data(match_id: str, cv_output: Dict) -> None:
    """
    Store processed match data in Supabase.
    
    Args:
        match_id: Match ID
        cv_output: Parsed CV output data
    """
    if not supabase:
        raise RuntimeError("Supabase not initialized")
    
    # Store match data JSON
    supabase.table("match_data").upsert({
        "match_id": match_id,
        "json_data": cv_output,
        "stats_summary": cv_output.get("stats", {}),
    }).execute()
    
    # Store individual shots
    shots = cv_output.get("shots", [])
    if shots:
        shots_to_insert = [
            {
                "match_id": match_id,
                "start_pos": shot["start_pos"],
                "end_pos": shot["end_pos"],
                "timestamp": shot.get("timestamp", 0),
                "video_timestamp": shot.get("video_timestamp"),
                "result": shot.get("result", "in_play"),
                "shot_type": shot.get("shot_type"),
            }
            for shot in shots
        ]
        
        supabase.table("shots").insert(shots_to_insert).execute()
    
    # Update match status
    supabase.table("matches").update({
        "status": "completed",
        "processed_at": "now()",
    }).eq("id", match_id).execute()


def trigger_processing(match_id: str) -> None:
    """
    Trigger video processing for a match.
    This would typically be called asynchronously (e.g., via Celery task).
    
    Args:
        match_id: Match ID to process
    """
    if not supabase:
        raise RuntimeError("Supabase not initialized")
    
    # Get match data
    match_response = supabase.table("matches").select("*").eq("id", match_id).single().execute()
    if not match_response.data:
        raise ValueError(f"Match {match_id} not found")
    
    match = match_response.data
    
    # Get player identification
    ident_response = supabase.table("player_identifications").select("*").eq("match_id", match_id).execute()
    identifications = ident_response.data or []
    
    if not identifications:
        raise ValueError("Player identification not found")
    
    # Update status to processing
    supabase.table("matches").update({
        "status": "processing",
    }).eq("id", match_id).execute()
    
    # TODO: Actually process video
    # For now, this is a placeholder
    # In production, this would:
    # 1. Download video from Playsight (if possible)
    # 2. Call CV backend
    # 3. Store results
    
    # Placeholder: Mark as completed with empty data
    # In real implementation, this would be done after actual processing
    # store_match_data(match_id, {"shots": [], "stats": {}})
