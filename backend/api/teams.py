from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
import os
from supabase import create_client, Client
import secrets
import string
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from auth import get_user_id

router = APIRouter(prefix="/api/teams", tags=["teams"])

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not supabase_url or not supabase_key:
    raise ValueError("Supabase credentials not configured")

supabase: Client = create_client(supabase_url, supabase_key)


class TeamCreate(BaseModel):
    name: str


class TeamJoin(BaseModel):
    code: str


def generate_team_code() -> str:
    """Generate a unique 6-character team code."""
    characters = string.ascii_uppercase + string.digits
    return ''.join(secrets.choice(characters) for _ in range(6))


@router.post("/create")
async def create_team(team_data: TeamCreate, user_id: str = Depends(get_user_id)):
    """
    Create a new team. Only coaches can create teams.
    Returns the team with the generated code.
    """
    # Verify user is a coach
    user_response = supabase.table("users").select("role").eq("id", user_id).single().execute()
    
    if not user_response.data or user_response.data.get("role") != "coach":
        raise HTTPException(status_code=403, detail="Only coaches can create teams")
    
    # Generate unique code
    code = generate_team_code()
    
    # Check if code already exists (unlikely but possible)
    existing = supabase.table("teams").select("id").eq("code", code).execute()
    if existing.data:
        code = generate_team_code()  # Try again
    
    # Create team
    team_response = supabase.table("teams").insert({
        "name": team_data.name,
        "code": code
    }).execute()
    
    if not team_response.data:
        raise HTTPException(status_code=500, detail="Failed to create team")
    
    team = team_response.data[0]
    team_id = team["id"]
    
    # Add coach to team_members
    supabase.table("team_members").insert({
        "team_id": team_id,
        "user_id": user_id,
        "role": "coach"
    }).execute()
    
    return {"team": team, "code": code}


@router.post("/join")
async def join_team(join_data: TeamJoin, user_id: str = Depends(get_user_id)):
    """
    Join a team using a team code. Any authenticated user can join.
    """
    # Look up team by code
    team_response = supabase.table("teams").select("*").eq("code", join_data.code).single().execute()
    
    if not team_response.data:
        raise HTTPException(status_code=404, detail="Invalid team code")
    
    team = team_response.data
    team_id = team["id"]
    
    # Check if user is already a member
    existing = supabase.table("team_members").select("id").eq("team_id", team_id).eq("user_id", user_id).execute()
    
    if existing.data:
        raise HTTPException(status_code=400, detail="Already a member of this team")
    
    # Add user to team
    member_response = supabase.table("team_members").insert({
        "team_id": team_id,
        "user_id": user_id,
        "role": "player"
    }).execute()
    
    if not member_response.data:
        raise HTTPException(status_code=500, detail="Failed to join team")
    
    return {"message": "Successfully joined team", "team": team}


@router.get("/my-teams")
async def get_my_teams(user_id: str = Depends(get_user_id)):
    """Get all teams the user belongs to."""
    # Get team memberships
    memberships = supabase.table("team_members").select("team_id").eq("user_id", user_id).execute()
    
    if not memberships.data:
        return {"teams": []}
    
    team_ids = [m["team_id"] for m in memberships.data]
    
    # Get team details
    teams_response = supabase.table("teams").select("*").in_("id", team_ids).execute()
    
    return {"teams": teams_response.data or []}


@router.get("/{team_id}/members")
async def get_team_members(team_id: str, user_id: str = Depends(get_user_id)):
    """Get all members of a team. User must be a member."""
    # Verify user is a member
    membership = supabase.table("team_members").select("role").eq("team_id", team_id).eq("user_id", user_id).execute()
    
    if not membership.data:
        raise HTTPException(status_code=403, detail="Not a member of this team")
    
    # Get all members
    members_response = supabase.table("team_members").select("*, users(*)").eq("team_id", team_id).execute()
    
    return {"members": members_response.data or []}
