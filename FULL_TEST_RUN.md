# Full Test Run Guide

## Prerequisites

✅ **Backend running** on http://localhost:8000  
✅ **Frontend running** on http://localhost:3000  
✅ **Supabase configured** with schema deployed

---

## Step-by-Step Test Flow

### 1. Create Your First Account (Coach)

1. **Go to** http://localhost:3000
2. **Click** "Don't have an account? Sign up"
3. **Fill in:**
   - Name: `Coach Test`
   - Email: `coach@test.com`
   - Password: `test123456` (min 6 characters)
4. **Click** "Sign up"

**Expected:**
- If email confirmations are **disabled** (recommended for dev): You'll be automatically signed in and redirected to `/dashboard`
- If email confirmations are **enabled**: You'll see an alert to check email. In that case, disable email confirmations in Supabase dashboard (Authentication → Settings → Email Auth → Toggle off "Enable email confirmations")

**What happens behind the scenes:**
- User is created in Supabase Auth
- Database trigger automatically creates profile in `public.users` table
- User role defaults to `'player'` (we'll change this to coach in next step)

---

### 2. Change Role to Coach (Manual Step)

Since there's no UI to change roles yet, you need to do this in Supabase:

1. **Go to** Supabase Dashboard → Table Editor → `users`
2. **Find** your user (coach@test.com)
3. **Click** the row to edit
4. **Change** `role` from `player` to `coach`
5. **Save**

**Alternative:** You can also do this via SQL:
```sql
UPDATE public.users 
SET role = 'coach' 
WHERE email = 'coach@test.com';
```

**Why?** Coaches can create teams, players can only join teams.

---

### 3. Test Dashboard

1. **Refresh** your dashboard page
2. **You should see:**
   - Welcome message with your name
   - User info (email, role)
   - Empty match list (or "No matches yet" message)
   - Left sidebar with: Dashboard, Stats, Teams, Logout
   - Bottom-right circular "+" button for uploading matches

**Try:**
- Click "Stats" in sidebar → Should show stats page (empty for now)
- Click "Teams" in sidebar → Should show teams page
- Click "Dashboard" → Should return to dashboard

---

### 4. Test Team Management (Coach)

1. **Go to** Teams page (click "Teams" in sidebar)
2. **You should see** "Create Team" section (because you're a coach)
3. **Enter** team name: `Varsity Team`
4. **Click** "Create Team"

**Expected:**
- Team is created
- You see a **team code** (e.g., `ABC123`)
- Team appears in "Your Teams" list
- Team code is displayed (copy this for next step)

**What happens:**
- Team is created in `public.teams` table
- You're automatically added as a coach member in `public.team_members`
- Unique code is generated (6 characters, alphanumeric)

---

### 5. Create a Player Account

1. **Log out** (click "Logout" in sidebar)
2. **Sign up** as a new user:
   - Name: `Player Test`
   - Email: `player@test.com`
   - Password: `test123456`
3. **Sign in** with player@test.com

**Note:** Player role is already set (defaults to 'player'), so no manual change needed.

---

### 6. Test Player Joining Team

1. **Go to** Teams page
2. **You should see** "Join Team" section (because you're a player)
3. **Enter** the team code from Step 4 (e.g., `ABC123`)
4. **Click** "Join Team"

**Expected:**
- Success message
- Team appears in "Your Teams" list
- You can see team members

**What happens:**
- Player is added to `public.team_members` table
- RLS policy allows players to join teams with valid codes

---

### 7. Test Video Upload Flow

1. **Go back to** Dashboard (as coach or player)
2. **Click** the circular "+" button (bottom right)
3. **Upload Modal opens:**
   - Enter a Playsight link: `https://playsight.com/video/12345` (or any test URL)
   - Click "Upload"

**Expected:**
- Modal closes
- Redirects to `/matches/[id]/identify` page
- Shows player identification interface

**What happens:**
- Match is created in `public.matches` table
- Status is set to `'processing'`
- Redirects to identification page

---

### 8. Test Player Identification

1. **On identification page**, you should see:
   - Match information
   - Placeholder frames (or actual frames if Playsight integration works)
   - Instructions to click on yourself

2. **Click** on yourself in each frame (or click anywhere for testing)
3. **Click** "Submit Identification"

**Expected:**
- Coordinates are saved to `public.player_identifications` table
- Redirects to match detail page: `/matches/[id]`

**Note:** This is a placeholder - actual frame extraction from Playsight isn't implemented yet.

---

### 9. Test Match Detail Page

1. **You should see:**
   - **Top:** Interactive tennis court diagram
   - **Side:** Video panel (empty until shot is clicked)
   - **Bottom:** Match stats section
   - **Status:** "Processing" indicator (if match isn't complete)

2. **Try clicking** on the court diagram:
   - If there are shots: Click on a shot line → Video panel should open
   - If no shots: Court should be empty (normal for new matches)

**What's working:**
- ✅ Court diagram renders
- ✅ Match stats display
- ✅ Processing status shows
- ⚠️ Shot data is placeholder (needs CV processing)
- ⚠️ Video embedding is placeholder (needs Playsight integration)

---

### 10. Test Stats Page

1. **Go to** Stats page (click "Stats" in sidebar)

**As Coach:**
- Should see list of all players on your teams
- Can select a player to see their stats
- Stats are aggregated from `public.shots` table

**As Player:**
- Should see your own stats
- Breakdown by game, set, match
- Stats are aggregated from your matches

**Note:** Stats will be empty until matches are processed and shots are added to the database.

---

### 11. Test Multiple Matches

1. **Upload another match** (click "+" button again)
2. **Go back to** Dashboard
3. **You should see:**
   - Multiple matches listed
   - Grouped by date (if matches are on different dates)
   - Each match is clickable

**As Coach:**
- Should see all matches from all team members
- Matches grouped by date with dropdowns

**As Player:**
- Should see only your own matches

---

## What's Real vs Placeholder

### ✅ **Fully Working:**
- User authentication (sign up, sign in, sign out)
- User profile creation (automatic via database trigger)
- Team creation (coaches)
- Team joining (players with codes)
- Team member listing
- Match creation
- Dashboard with match listings
- Match detail page structure
- Court diagram rendering
- Stats page structure
- Navigation and routing
- Protected routes (middleware)
- Backend API endpoints (all functional)

### ⚠️ **Placeholder / Needs Implementation:**
- **Playsight frame extraction** - Currently just accepts link, doesn't extract frames
- **Player tracking** - Color recognition not implemented
- **CV processing** - Backend service exists but doesn't call actual CV pipeline
- **Shot data** - Court diagram works but needs real shot data from CV processing
- **Video embedding** - Video panel exists but needs Playsight embed URL
- **Real-time processing updates** - Structure exists but needs actual processing pipeline
- **Stats aggregation** - Works but needs shot data to be meaningful

---

## Testing Backend API Directly

You can also test the backend API using Swagger UI:

1. **Go to** http://localhost:8000/docs
2. **Click** "Authorize" button (top right)
3. **Get your JWT token:**
   - In browser console on frontend (after logging in):
     ```javascript
     const { data } = await supabase.auth.getSession()
     console.log(data.session.access_token)
     ```
   - Copy the token
4. **Paste token** in Swagger UI authorization (format: `Bearer YOUR_TOKEN`)
5. **Test endpoints:**
   - `GET /api/teams` - List your teams
   - `POST /api/teams` - Create a team
   - `GET /api/matches` - List your matches
   - `POST /api/matches` - Create a match
   - `GET /api/stats/player/{player_id}` - Get player stats

---

## Common Issues & Solutions

### "Email confirmation required"
**Solution:** Disable email confirmations in Supabase:
- Dashboard → Authentication → Settings → Email Auth
- Toggle off "Enable email confirmations"

### "Can't create team" (as player)
**Solution:** Make sure your role is set to `'coach'` in `public.users` table

### "Can't join team" (invalid code)
**Solution:** 
- Check the team code is correct (case-sensitive)
- Verify team exists in `public.teams` table
- Check RLS policies are enabled

### "No matches showing"
**Solution:**
- Create a match via the "+" button
- Check `public.matches` table in Supabase
- Verify you're logged in as the correct user

### "Stats are empty"
**Solution:** This is normal - stats populate when matches are processed and shots are added. For testing, you can manually add shot data to `public.shots` table.

---

## Next Steps After Testing

Once you've verified everything works:

1. **Implement Playsight integration** - Extract frames from Playsight videos
2. **Connect CV backend** - Call actual CV processing pipeline
3. **Implement player tracking** - Color recognition and tracking
4. **Add real shot data** - Populate shots table from CV output
5. **Complete video embedding** - Generate Playsight embed URLs
6. **Add error handling** - Better user feedback
7. **Add loading states** - Show spinners during API calls

---

**Happy Testing! 🎾**
