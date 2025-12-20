# Tennis Analytics Application

A full-stack tennis analytics web application for coaches and players to track match performance, visualize shot patterns, and analyze statistics.

## Tech Stack

- **Frontend**: Next.js 14+ (App Router), TypeScript, Tailwind CSS, shadcn/ui
- **Backend**: FastAPI (Python)
- **Database & Auth**: Supabase (PostgreSQL + Auth + Realtime)
- **State Management**: TanStack Query, Zustand
- **Forms**: React Hook Form + Zod
- **Charts**: Recharts

## Prerequisites

- Node.js 18+ and npm
- Python 3.8+
- Supabase account (free tier works)

## Complete Setup Guide

Follow these steps in order to get everything running.

### Step 1: Create Supabase Project

1. Go to [https://supabase.com](https://supabase.com) and sign up/login
2. Click **"New Project"**
3. Fill in:
   - **Project Name**: `tennis-analytics` (or your preferred name)
   - **Database Password**: Create a strong password (save this!)
   - **Region**: Choose closest to you
   - **Pricing Plan**: Free tier is fine for development
4. Wait for project to initialize (takes ~2 minutes)

### Step 2: Get Supabase Credentials

1. In your Supabase project dashboard, go to **Settings** → **API**
2. Copy the following (you'll need these in Step 4):
   - **Project URL** (e.g., `https://xxxxx.supabase.co`)
   - **anon/public key** (starts with `eyJ...`)
   - **service_role key** (keep this secret! Only for backend)

### Step 3: Set Up Database Schema

1. **Open SQL Editor**
   - In Supabase dashboard, click **SQL Editor** in the left sidebar
   - Click **New Query** button

2. **Run the Schema**
   - Open `supabase/schema.sql` file from this project
   - Copy the **entire contents** of the file
   - Paste into the SQL Editor
   - Click **Run** (or press Cmd/Ctrl + Enter)
   - You should see: **"Success. No rows returned"** (this is normal for DDL statements)

3. **Verify Tables Created**
   - Go to **Table Editor** in the left sidebar
   - You should see these tables:
     - ✅ `users` - User profiles (extends auth.users)
     - ✅ `teams` - Teams with unique codes
     - ✅ `team_members` - Junction table for team membership
     - ✅ `matches` - Match records
     - ✅ `match_data` - JSON data from CV processing
     - ✅ `shots` - Individual shot records
     - ✅ `player_identifications` - Player identification data

**Note**: If you see warnings about `auth.users` or `uuid-ossp`, that's normal - ignore them. The "Success. No rows returned" message is expected for schema creation.

### Step 4: Configure Environment Variables

**Frontend** (`frontend/.env.local`):
```env
NEXT_PUBLIC_SUPABASE_URL=your_project_url_here
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key_here
NEXT_PUBLIC_API_URL=http://localhost:8000
```

**Backend** (`backend/.env`):
```env
SUPABASE_URL=your_project_url_here
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key_here
SUPABASE_ANON_KEY=your_anon_key_here
ALLOWED_ORIGINS=http://localhost:3000
API_PORT=8000
ENVIRONMENT=development
```

Replace the placeholder values with your actual Supabase credentials from Step 2.

### Step 5: Install Dependencies

**Frontend:**
```bash
cd frontend
npm install
```

**Backend:**
```bash
cd backend
python3 -m venv ../tennis_env  # Create venv in project root
source ../tennis_env/bin/activate  # On Windows: ..\tennis_env\Scripts\activate
pip install -r requirements.txt
```

### Step 6: Configure Email Authentication (Recommended for Development)

Email/password authentication is enabled by default in Supabase. For easier development, disable email confirmation:

1. Go to **Authentication** → **Settings** in Supabase dashboard
2. Under "Email Auth", toggle off **"Enable email confirmations"**
3. (Re-enable for production!)

This lets you sign up and immediately sign in without checking email. The database trigger will automatically create a user profile when someone signs up.

### Step 7: Run Development Servers

**Option A: Using Helper Scripts (Easiest)**

**Terminal 1 - Frontend:**
```bash
./start_frontend.sh
```
Frontend will be available at http://localhost:3000

**Terminal 2 - Backend:**
```bash
./start_backend.sh
```
Backend API will be available at http://localhost:8000  
API docs at http://localhost:8000/docs

**Option B: Manual Commands**

**Terminal 1 - Frontend:**
```bash
cd frontend
npm run dev
```
Frontend will be available at http://localhost:3000

**Terminal 2 - Backend:**
```bash
cd backend
source ../tennis_env/bin/activate  # On Windows: ..\tennis_env\Scripts\activate
uvicorn main:app --reload --port 8000
```
Backend API will be available at http://localhost:8000  
API docs at http://localhost:8000/docs

### Step 8: Test the Setup

1. **Visit the App**
   - Go to http://localhost:3000
   - You should be redirected to `/login`

2. **Create a Test Account**
   - Click "Don't have an account? Sign up"
   - Enter:
     - Name: Your name
     - Email: your-email@example.com
     - Password: (at least 6 characters)
   - Click "Sign up"

3. **Sign In**
   - Use your email and password
   - You should be redirected to `/dashboard`
   - You should see your user info

If everything works, you're ready to start building! 🎾

## Project Structure

```
tennis_analytics/
├── frontend/              # Next.js frontend
│   ├── app/              # App Router pages
│   │   ├── login/        # Login/signup page
│   │   └── dashboard/    # Dashboard page
│   ├── components/        # React components
│   ├── lib/              # Utilities & Supabase client
│   │   └── supabase/     # Supabase client config
│   └── hooks/            # Custom React hooks
│       └── useAuth.ts    # Authentication hook
├── backend/              # FastAPI backend
│   ├── api/              # API route handlers
│   ├── services/         # Business logic
│   ├── models/           # Data models
│   └── main.py           # FastAPI app entry point
├── supabase/             # Database schema
│   └── schema.sql        # Complete database schema
└── old/                  # Legacy CV backend code
```

## What's Already Set Up

✅ **Project Structure**: Frontend and backend directories created  
✅ **Database Schema**: All tables created with relationships and RLS policies (users, teams, team_members, matches, match_data, shots, player_identifications)  
✅ **Supabase Configuration**: Schema successfully deployed, tables verified  
✅ **Authentication**: Email/password auth with Supabase (simplified, no OAuth needed)  
✅ **Auto User Profile**: Database trigger automatically creates user profile on signup  
✅ **Login/Signup Pages**: Fully functional at `/login`  
✅ **Protected Routes**: Dashboard requires authentication, middleware configured  
✅ **Frontend Structure**: Next.js 14+ with TypeScript, Tailwind CSS, and shadcn/ui  
✅ **Supabase Client**: Client and server-side Supabase configuration ready  
✅ **Backend Structure**: FastAPI initialized with CORS configured  
✅ **Team Management Schema**: Supports coaches creating teams with codes, players joining via codes  

## Troubleshooting

### Frontend won't start
- Make sure Node.js 18+ is installed: `node --version`
- Check that `frontend/.env.local` has valid Supabase credentials
- Try: `cd frontend && npm install` again

### Backend won't start
- Make sure Python 3.8+ is installed: `python3 --version`
- Activate virtual environment: `source tennis_env/bin/activate` (from project root)
- Install dependencies: `pip install -r backend/requirements.txt`
- Check `backend/.env` file has valid Supabase credentials

### Database errors
- Make sure you've run `supabase/schema.sql` in Supabase SQL Editor
- Check that all tables are created in Table Editor
- Verify your Supabase credentials in `.env` files are correct

### Can't sign up / Email not sending
- Check **Authentication** → **Settings** → **Email Auth** in Supabase
- For development, disable email confirmation (see Step 6)
- Check Supabase project is active (not paused)

### User profile not created
- Check the database trigger was created
- In Supabase SQL Editor, run: `SELECT * FROM public.users;`
- If empty, re-run the schema.sql file

### "relation auth.users does not exist" warning
- This is normal! `auth.users` is created automatically by Supabase
- The schema references it, but you don't create it manually
- You can safely ignore this warning

## Development

### Frontend Development

- Uses Next.js 14+ App Router
- Pages in `app/` directory
- Components in `components/` (create as needed)
- Hooks in `hooks/`
- Supabase client in `lib/supabase/`

### Backend Development

- FastAPI application in `backend/main.py`
- Add API routes in `backend/api/`
- Add services in `backend/services/`
- Add models in `backend/models/`

### Database Changes

- Schema defined in `supabase/schema.sql`
- Run SQL directly in Supabase SQL Editor
- Row Level Security (RLS) policies are configured

## Current Status

### ✅ Completed Features

**Phase 1: Foundation**
- ✅ Project structure (frontend/backend directories)
- ✅ Next.js 14+ frontend with TypeScript, Tailwind CSS, shadcn/ui
- ✅ FastAPI backend structure
- ✅ Supabase database schema deployed and verified
- ✅ Email/password authentication working
- ✅ Login/signup pages functional

**Phase 2: Core UI**
- ✅ Main layout with sidebar navigation and floating action button
- ✅ Dashboard page with date-organized match listings
- ✅ Stats page structure (coach/player views)
- ✅ Responsive design

**Phase 3: Match Visualization**
- ✅ Interactive tennis court diagram component
- ✅ Shot visualization with color coding (winners/errors/in-play)
- ✅ Clickable shot lines with hover states
- ✅ Video panel for Playsight embedding
- ✅ Match detail page with court and stats
- ✅ Match stats display component

**Phase 4: Video Processing**
- ✅ Upload modal with Playsight link input
- ✅ Player identification interface (multi-frame click-to-identify)
- ✅ Processing status component with real-time updates
- ✅ Backend endpoints for video upload and processing

**Phase 5: Team Management**
- ✅ Team creation (coaches) with code generation
- ✅ Player joining via team codes
- ✅ Team members display
- ✅ Teams page with coach/player views
- ✅ Backend API for team management

**Phase 6: Backend API**
- ✅ FastAPI endpoints for teams (create, join, list, members)
- ✅ FastAPI endpoints for matches (list, get, create)
- ✅ FastAPI endpoints for videos (upload, identify player, status)
- ✅ FastAPI endpoints for stats (player stats, season stats)
- ✅ Authentication middleware with Supabase token verification

**Phase 7: Integration**
- ✅ CV backend integration service structure
- ✅ Player tracking service (color recognition)
- ✅ Playsight integration research and placeholder

### 🚧 Still Needs Implementation

1. **Playsight Frame Extraction** - Backend needs to extract frames from Playsight videos for player identification
2. **CV Backend Integration** - Connect to actual CV processing pipeline in `old/src/core/tennis_CV.py`
3. **Player Tracking** - Implement actual color recognition and player tracking throughout video
4. **Stats Visualization** - Add Recharts components for detailed statistics display
5. **Video Processing Workflow** - Complete the async video processing pipeline
6. **Error Handling** - Add comprehensive error handling and user feedback
7. **Loading States** - Add loading indicators throughout the app
8. **Testing** - Add unit tests and integration tests

## Features Status

- [x] **User authentication** - Email/password signup and login working
- [x] **Database schema** - All tables and RLS policies configured
- [x] **Team management** - Full UI and backend API for coaches creating teams and players joining
- [x] **Dashboard** - Match listings with date organization
- [x] **Match upload** - Playsight link submission UI
- [x] **Interactive court visualization** - Court diagram with shot patterns
- [x] **Shot visualization** - Clickable shots with color coding
- [x] **Video panel** - Side panel for Playsight video embedding
- [x] **Player identification** - Multi-frame click-to-identify interface
- [x] **Processing status** - Real-time status updates via Supabase Realtime
- [x] **Backend API** - All endpoints for teams, matches, videos, stats
- [ ] **Playsight frame extraction** - Extract frames from Playsight videos
- [ ] **CV processing** - Connect to actual CV backend for video analysis
- [ ] **Statistics visualization** - Charts and detailed stats display
- [ ] **Player tracking** - Implement color recognition tracking

## Need Help?

- Check Supabase dashboard for database issues
- Check browser console for frontend errors
- Check terminal output for backend errors
- Verify all environment variables are set correctly

---

## Large Files & Git Notes

### ⚠️ Important: Large Model File

**`SAM3/model.safetensors` (3.2GB)** - This file is **excluded** from git because it exceeds GitHub's 100MB file limit.

**To use SAM3 features, download the model separately:**
- See `SAM3/README_DOWNLOAD.md` for download instructions
- Or run: `huggingface-cli download facebook/sam3 --local-dir ./SAM3`

### Files Excluded from Git

The following are automatically excluded via `.gitignore`:

**Model Files:**
- `*.safetensors` - All safetensors files (including SAM3 model)
- `*.pt`, `*.pth` - PyTorch model files
- `*.cbm` - CatBoost model files

**Dependencies:**
- `tennis_env/` - Python virtual environment (create locally: `python3 -m venv tennis_env`)
- `frontend/node_modules/` - Node dependencies (install with `npm install`)

**Media & Data:**
- Video files (`.mp4`, `.avi`, `.mov`, etc.)
- Test videos in `tests/` directory
- Image files (`.jpg`, `.png`, etc.)

**Build Outputs:**
- `frontend/.next/` - Next.js build output
- `frontend/out/` - Next.js export output

**Environment Files:**
- `.env`, `.env.local` - Sensitive credentials (never committed)

### Ready to Push

All sensitive files and large files are properly excluded. The codebase is ready for GitHub!

```bash
# Check what will be committed
git status

# Verify large files are ignored
git check-ignore SAM3/model.safetensors

# Add and commit
git add .
git commit -m "Tennis Analytics: Complete frontend and backend implementation"

# Push to GitHub
git push origin main
```

---

**Ready to push!** 🚀
