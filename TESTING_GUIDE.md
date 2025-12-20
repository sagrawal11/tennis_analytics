# Testing Guide

## Quick Start

### 1. Start Backend Server

Open Terminal 1:
```bash
cd "/Users/sarthak/Desktop/App Projects/tennis_analytics/backend"
source ../tennis_env/bin/activate
uvicorn main:app --reload --port 8000
```

**Expected output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

**Test backend:**
- Visit http://localhost:8000/docs - Should see FastAPI Swagger UI
- Visit http://localhost:8000/api/health - Should return `{"status": "ok"}` (if endpoint exists)

### 2. Start Frontend Server

Open Terminal 2:
```bash
cd "/Users/sarthak/Desktop/App Projects/tennis_analytics/frontend"
npm run dev
```

**Expected output:**
```
  ▲ Next.js 14.x.x
  - Local:        http://localhost:3000
  - Ready in Xms
```

**Test frontend:**
- Visit http://localhost:3000 - Should redirect to `/login`

## Testing Checklist

### ✅ Authentication Flow

1. **Sign Up**
   - Go to http://localhost:3000/login
   - Click "Don't have an account? Sign up"
   - Fill in:
     - Name: Test User
     - Email: test@example.com
     - Password: test123456
   - Click "Sign up"
   - Should redirect to dashboard

2. **Sign In**
   - Log out (click logout in sidebar)
   - Sign in with test@example.com / test123456
   - Should redirect to dashboard

### ✅ Dashboard

1. **View Dashboard**
   - After login, should see dashboard at http://localhost:3000/dashboard
   - Should see "Welcome" message and user info
   - Should see empty match list (or existing matches if any)

2. **Navigation**
   - Click "Dashboard" in sidebar - should navigate to dashboard
   - Click "Stats" in sidebar - should navigate to stats page
   - Click "Teams" in sidebar - should navigate to teams page
   - Click "Logout" - should sign out and redirect to login

### ✅ Teams Management

**As Coach:**
1. Go to Teams page
2. Click "Create Team"
3. Enter team name (e.g., "Varsity Team")
4. Should see team code generated
5. Should see team in "Your Teams" list

**As Player:**
1. Sign up as a player (or use existing player account)
2. Go to Teams page
3. Enter team code from coach
4. Click "Join Team"
5. Should see team in "Your Teams" list

### ✅ Video Upload Flow

1. **Upload Match**
   - Click the "+" button (bottom right)
   - Enter a Playsight link (e.g., `https://playsight.com/video/12345`)
   - Click "Upload"
   - Should redirect to player identification page

2. **Player Identification**
   - Should see placeholder frames
   - Click on yourself in each frame
   - Click "Submit Identification"
   - Should redirect to match detail page

3. **Processing Status**
   - Match detail page should show "Processing" status
   - Should see progress updates (if real-time is working)

### ✅ Match Detail Page

1. **View Match**
   - Click on a match from dashboard
   - Should see:
     - Court diagram (top)
     - Video panel (side, if shot selected)
     - Match stats (bottom)

2. **Court Interaction**
   - Hover over shot lines - should highlight
   - Click on shot line - should open video panel
   - Video panel should show Playsight embed

### ✅ Stats Page

1. **View Stats**
   - Go to Stats page
   - Should see stats overview (coach sees all players, player sees own stats)
   - Should show breakdown by game/set/match

## Troubleshooting

### Backend won't start
- Check if port 8000 is already in use: `lsof -i :8000`
- Verify virtual environment is activated: `which python` should show `tennis_env`
- Check backend/.env has valid Supabase credentials
- Check backend logs for errors

### Frontend won't start
- Check if port 3000 is already in use: `lsof -i :3000`
- Verify node_modules installed: `ls frontend/node_modules`
- Check frontend/.env.local has valid Supabase credentials
- Check browser console for errors (F12)

### Authentication errors
- Verify Supabase project is active
- Check Supabase credentials in .env files
- Check Supabase dashboard → Authentication → Settings
- Verify email confirmations are disabled (for dev)

### Database errors
- Check Supabase dashboard → SQL Editor
- Verify schema.sql was run successfully
- Check RLS policies are enabled
- Verify user profile was created (check `public.users` table)

### CORS errors
- Verify `ALLOWED_ORIGINS=http://localhost:3000` in backend/.env
- Check backend/main.py has CORS middleware configured
- Restart backend server after changing .env

## API Testing

### Using Swagger UI (Recommended)

1. Start backend server
2. Visit http://localhost:8000/docs
3. Click "Authorize" button
4. Get token from browser console after login:
   ```javascript
   // In browser console on frontend
   const { data } = await supabase.auth.getSession()
   console.log(data.session.access_token)
   ```
5. Paste token in Swagger UI authorization
6. Test endpoints directly

### Using curl

```bash
# Get auth token (after logging in via frontend)
TOKEN="your_jwt_token_here"

# Test teams endpoint
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/teams

# Test matches endpoint
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/matches
```

## Next Steps

Once basic testing passes:
1. Test team creation and joining
2. Test match upload workflow
3. Test court visualization with sample data
4. Test stats aggregation
5. Test real-time processing updates
