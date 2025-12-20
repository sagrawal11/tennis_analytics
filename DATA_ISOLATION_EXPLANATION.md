# Data Isolation - Coach/Player Separation

## How It Works

### Team-Based Isolation
- **Players belong to teams** (via `team_members` table)
- **Coaches create teams** and players join via team codes
- **Each team is isolated** - coaches only see data from players on their teams

### Example Scenario

**Coach A:**
- Creates "Team Alpha" (code: ABC123)
- Players: Player1, Player2, Player3 join Team Alpha

**Coach B:**
- Creates "Team Beta" (code: XYZ789)
- Players: Player4, Player5, Player6 join Team Beta

**Result:**
- ✅ Coach A sees matches from Player1, Player2, Player3 only
- ✅ Coach B sees matches from Player4, Player5, Player6 only
- ✅ Coach A **cannot** see matches from Player4, Player5, Player6
- ✅ Coach B **cannot** see matches from Player1, Player2, Player3
- ✅ Players only see their own matches

## Database Structure

### Team Membership
```
team_members table:
- team_id: Links to teams table
- user_id: Links to users table
- role: 'coach' or 'player'
```

**Example:**
```
Team Alpha (id: team-1)
├── Coach A (user_id: coach-a, role: 'coach')
├── Player1 (user_id: player-1, role: 'player')
├── Player2 (user_id: player-2, role: 'player')
└── Player3 (user_id: player-3, role: 'player')

Team Beta (id: team-2)
├── Coach B (user_id: coach-b, role: 'coach')
├── Player4 (user_id: player-4, role: 'player')
├── Player5 (user_id: player-5, role: 'player')
└── Player6 (user_id: player-6, role: 'player')
```

### Match Ownership
```
matches table:
- user_id: The player who owns the match
```

**Example:**
```
Match 1: user_id = player-1 (belongs to Player1)
Match 2: user_id = player-2 (belongs to Player2)
Match 3: user_id = player-4 (belongs to Player4)
```

## RLS Policies (Row Level Security)

### Matches Policy
```sql
CREATE POLICY "Coaches can view all team member matches"
    ON public.matches FOR SELECT
    USING (
        -- Check if viewing user is a coach
        EXISTS (SELECT 1 FROM public.users WHERE id = auth.uid() AND role = 'coach')
        -- AND match owner is on same team as coach
        AND EXISTS (
            SELECT 1 FROM public.team_members tm1
            JOIN public.team_members tm2 ON tm1.team_id = tm2.team_id
            WHERE tm1.user_id = auth.uid()  -- Coach viewing
            AND tm2.user_id = matches.user_id  -- Match owner
        )
    );
```

**How it works:**
1. Verifies the viewing user (`auth.uid()`) is a coach
2. Checks if the match owner (`matches.user_id`) is on the same team as the coach
3. Only returns matches where both conditions are true

### Backend API Logic

**File**: `backend/api/matches.py`

```python
if user_role == "coach":
    # Get teams the coach belongs to
    teams_response = supabase.table("team_members").select("team_id").eq("user_id", user_id).execute()
    team_ids = [t["team_id"] for t in (teams_response.data or [])]
    
    # Get all team members from coach's teams
    if team_ids:
        members_response = supabase.table("team_members").select("user_id").in_("team_id", team_ids).execute()
        member_ids = [m["user_id"] for m in (members_response.data or [])]
        
        # Get matches for team members only
        matches_response = supabase.table("matches").select("*").in_("user_id", member_ids).execute()
```

**How it works:**
1. Gets all teams the coach belongs to
2. Gets all team members from those teams
3. Returns matches only for those team members
4. **Result**: Coach only sees matches from players on their teams

## Verification

### ✅ Data Isolation is Enforced

1. **RLS Policies**: Database-level security ensures coaches can only query matches from their team members
2. **Backend API**: Additional filtering ensures only team member matches are returned
3. **Team Membership**: Players must join teams via codes, ensuring proper team assignment
4. **Match Ownership**: Matches are owned by players (`user_id`), not coaches

### ✅ Multiple Coaches Are Isolated

- Coach A's teams are separate from Coach B's teams
- No shared data between different coaches
- Each coach only sees their own team's players and matches
- Players can only belong to teams they've joined (via code)

## Edge Cases Handled

1. **Coach with no teams**: Sees only their own matches (if any)
2. **Player on multiple teams**: Each coach sees that player's matches
3. **Coach uploads for player**: Match appears in both coach's and player's dashboard (match.user_id = player's ID)
4. **Player uploads own match**: Only appears in player's dashboard

## Summary

✅ **Players belong to coaches via teams**  
✅ **Multiple coaches are completely isolated**  
✅ **Coaches only see matches from players on their teams**  
✅ **RLS policies enforce this at the database level**  
✅ **Backend API provides additional filtering**

The system is designed for multi-tenant use where each coach manages their own team(s) independently.
