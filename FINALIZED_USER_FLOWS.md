# Finalized User Flows - Complete Reference

This document summarizes all finalized user flows and decisions for the Tennis Analytics application.

## Key Decisions Summary

### 1. Role Assignment
- **During Signup**: Users select role (Coach/Player) via radio buttons
- **Default**: Player
- **Database**: Role stored in `users.role` field

### 2. Match Upload
- **Players**: Upload their own matches
- **Coaches**: Can upload matches for any team member
  - Select player from dropdown before upload
  - Match appears in BOTH coach's and player's dashboard
  - Match `user_id` = selected player's ID

### 3. Dashboard Organization
- **Players**: Matches grouped by date only
- **Coaches**: 
  - Matches grouped by date
  - Player filter dropdown at top
  - Filter: "All Players" or specific player name
  - When filtered: Shows only that player's matches

### 4. Team Code Sharing
- Display code prominently after creation
- No copy button (players type code manually)
- Code shown in large, readable format

### 5. Stats Page
- **Layout**: Summary cards (top) + Detailed charts (below)
- **Coach View**: Player selector dropdown, then stats
- **Player View**: Own stats only

### 6. Player Identification
- **5 frames** (updated from 3)
- Grid layout: 3 columns on desktop, responsive
- Click on yourself in each frame

### 7. Empty States
- **Onboarding messages** instead of simple empty states
- Welcome messages with next steps
- Contextual help text

### 8. Processing Status
- **Wait until complete**: No partial data shown
- Court diagram only appears when `status === 'completed'`
- Clear status messaging

### 9. Navigation
- **Profile icon** in sidebar header (replaces sign out button)
- **Dropdown menu**: Profile option + Sign Out option
- Profile page at `/profile`

### 10. Profile Page
- **Basic info**: Name, email, role, teams
- **View-only** for now
- Clean card layout

## Complete User Flows

### Flow 1: Sign Up (Coach)
1. Visit `/login`
2. Click "Don't have an account? Sign up"
3. **Select role**: Choose "Coach" (radio button)
4. Fill in: Name, Email, Password
5. Click "Sign up"
6. Auto-redirect to `/dashboard` (if email confirmations disabled)
7. Role saved to database

### Flow 2: Sign Up (Player)
1. Visit `/login`
2. Click "Don't have an account? Sign up"
3. **Select role**: Choose "Player" (default, or explicitly select)
4. Fill in: Name, Email, Password
5. Click "Sign up"
6. Auto-redirect to `/dashboard`
7. Role saved to database

### Flow 3: Coach Creates Team
1. Go to `/teams`
2. See onboarding message if no teams: "Welcome! Get started by creating your first team."
3. Click "Create New Team"
4. Enter team name
5. Click "Create Team"
6. See success message with team code
7. Code displayed prominently (large, readable)
8. Team appears in "Your Teams" list

### Flow 4: Player Joins Team
1. Go to `/teams`
2. See onboarding message if no teams: "Join a team to get started!"
3. Enter team code (auto-uppercased)
4. Click "Join"
5. On success: Team appears in "Your Teams" list
6. On error: Error message displayed

### Flow 5: Player Uploads Own Match
1. Click floating "+" button
2. Modal opens
3. Enter Playsight link
4. (Optional) Enter player name
5. Click "Upload"
6. Redirect to `/matches/[id]/identify`
7. Click on yourself in 5 frames
8. Submit identification
9. Redirect to match detail page
10. Match appears in player's dashboard

### Flow 6: Coach Uploads Match for Player
1. Click floating "+" button
2. Modal opens
3. **Select player** from dropdown (team members list)
4. Enter Playsight link
5. Click "Upload"
6. Redirect to `/matches/[id]/identify`
7. Coach clicks on player in 5 frames
8. Submit identification
9. Redirect to match detail page
10. **Match appears in BOTH coach's and player's dashboard**

### Flow 7: Coach Views Dashboard with Filter
1. Visit `/dashboard`
2. See player filter dropdown at top
3. Default: "All Players" (shows all team matches)
4. Select specific player from dropdown
5. Dashboard updates: Shows only that player's matches
6. Still grouped by date
7. Can switch back to "All Players"

### Flow 8: View Match (Processing)
1. Click match card from dashboard
2. Go to `/matches/[id]`
3. See processing status indicator
4. **No court diagram** (status !== 'completed')
5. Message: "Processing in progress. Court visualization will appear when processing is complete."

### Flow 9: View Match (Completed)
1. Click match card from dashboard
2. Go to `/matches/[id]`
3. See court diagram with shots
4. Click on shot line
5. Video panel opens with Playsight embed
6. Video jumps to shot timestamp
7. Can close video panel
8. See match stats below court

### Flow 10: Access Profile
1. Click profile icon in sidebar header
2. Dropdown menu appears
3. Click "Profile"
4. Navigate to `/profile`
5. See profile information:
   - Name
   - Email
   - Role (badge)
   - Teams list

### Flow 11: Sign Out
1. Click profile icon in sidebar header
2. Dropdown menu appears
3. Click "Sign Out"
4. Sign out and redirect to `/login`

### Flow 12: View Stats (Coach)
1. Go to `/stats`
2. See player selector dropdown
3. Select player (or "All Players")
4. See summary cards:
   - Total Matches
   - Total Shots
   - Win Rate
   - Average Shots per Match
5. See detailed charts:
   - Winners/Errors Over Time
   - Shot Distribution
   - Performance by Match
   - Shot Patterns

### Flow 13: View Stats (Player)
1. Go to `/stats`
2. See own summary cards (same as coach view)
3. See own detailed charts (same as coach view)
4. No player selector (only own stats)

## Database Schema Notes

### Current Schema Supports:
- ✅ User roles (coach/player)
- ✅ Team membership
- ✅ Match ownership (user_id)
- ✅ Team members relationship
- ✅ Profile data (name, email, role, teams via joins)

### No Schema Changes Needed:
- All profile data can be queried from existing tables
- Teams can be fetched via `team_members` join
- Match ownership handled via `matches.user_id`

## API Endpoints Used

### Teams
- `GET /api/teams/my-teams` - Get user's teams
- `POST /api/teams/create` - Create team (coaches)
- `POST /api/teams/join` - Join team (players)
- `GET /api/teams/{team_id}/members` - Get team members (for player dropdown)

### Matches
- `GET /api/matches` - Get user's matches (coaches see all team matches)
- `POST /api/matches` - Create match (with user_id for coach uploads)
- `GET /api/matches/{id}` - Get match details

### Videos
- `POST /api/videos/identify-player` - Submit player identification

### Stats
- `GET /api/stats/player/{player_id}` - Get player stats

## Integration Points

### Signup Form
- Add role selection UI
- Pass role to `signUp(email, password, name, role)`
- Update `useAuth` hook to accept role parameter

### Upload Modal
- Add player dropdown (coaches only)
- Fetch team members from API
- Pass `user_id` (selected player) to match creation

### Dashboard
- Add player filter dropdown (coaches only)
- Fetch team members for filter options
- Filter matches array by selected player

### Sidebar
- Replace "Sign Out" button with profile icon
- Add dropdown menu component
- Handle navigation to profile page

### Profile Page
- New route: `/app/profile/page.tsx`
- Fetch user data and teams
- Display in card layout

### Match Detail
- Conditional rendering: Only show court diagram if `status === 'completed'`
- Show processing status if not completed

---

**This document serves as the complete reference for all user flows and decisions.**
