# Courtvision - Computer Vision Powered Tennis Analytics

A full-stack tennis analytics web application for coaches and players to track match performance, visualize shot patterns, and analyze statistics using computer vision. Features team-based activation keys for easy access management.

## 🎾 Overview

Courtvision enables tennis coaches to manage teams, track player performance, and analyze match data through an intuitive web interface. Players can upload their matches, view their shot patterns on an interactive court diagram, and track their statistics over time.

### Key Features

- **Role-Based Access**: Coaches and players with different permissions and views
- **Activation Key System**: Team-based activation - one key activates entire team
- **Team Management**: Coaches create teams with codes; players join via codes
- **Match Upload**: Upload Playsight video links for processing
- **Interactive Court Visualization**: Clickable shot patterns on a tennis court diagram
- **Real-Time Processing**: Track video processing status with live updates
- **Statistics Tracking**: Comprehensive stats for players and teams
- **Modern UI**: Beautiful landing page, dark theme with emerald green (#50C878) accents

---

## 🛠 Tech Stack

### Frontend
- **Framework**: Next.js 14+ (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS + shadcn/ui components
- **State Management**: TanStack Query (React Query) for server state
- **Forms**: React Hook Form + Zod validation
- **Charts**: Recharts (for future stats visualization)
- **Icons**: Lucide React

### Backend
- **Framework**: FastAPI (Python)
- **Database Client**: Supabase Python client
- **Authentication**: Supabase Auth (JWT tokens)

### Database & Infrastructure
- **Database**: Supabase (PostgreSQL)
- **Authentication**: Supabase Auth (email/password)
- **Real-time**: Supabase Realtime (for processing status updates)
- **Row Level Security**: RLS policies for data isolation

### Deployment
- **Frontend**: Vercel (planned)
- **Backend**: Railway (planned)
- **Current**: Local development

---

## ✅ Completed Features

### Phase 1: Foundation & Authentication
- ✅ Project structure (frontend/backend directories)
- ✅ Next.js 14+ frontend with TypeScript, Tailwind CSS, shadcn/ui
- ✅ FastAPI backend structure with CORS configuration
- ✅ Supabase database schema deployed and verified
- ✅ Email/password authentication working
- ✅ Landing page with hero section, features, testimonials
- ✅ Modal-based authentication (sign in/sign up)
- ✅ Role-based signup (coach/player selection)
- ✅ Database trigger automatically creates user profile on signup
- ✅ Protected routes with middleware
- ✅ Profile page with user information

### Phase 2: Core UI & Navigation
- ✅ Main layout with sidebar navigation
- ✅ Floating action button for quick actions
- ✅ Dashboard page with date-organized match listings
- ✅ Stats page structure (coach/player views)
- ✅ Teams page with coach/player views
- ✅ Profile page with user info and teams
- ✅ Responsive design (mobile-friendly)
- ✅ Dark theme with emerald green (#50C878) accents
- ✅ Cursor pointer on all clickable elements

### Phase 3: Team Management
- ✅ Team creation (coaches) with unique code generation
- ✅ Team joining via codes (players and coaches)
- ✅ Team members display (shows all coaches and players)
- ✅ Teams page with separate views for coaches and players
- ✅ Backend API for team management (create, join, list, members)
- ✅ Coaches can join existing teams (multi-coach support)
- ✅ Players can only join teams
- ✅ Team members list with role badges

### Phase 3.5: Activation Key System
- ✅ Activation key support in database schema
- ✅ Coach activation via activation keys
- ✅ Team-based activation sharing (one key activates entire team)
- ✅ Auto-activation when joining activated teams
- ✅ Activation key input component for coaches
- ✅ Locked states for unactivated coaches (can't create teams or upload)
- ✅ Locked states for players without activated teams (can't upload)
- ✅ Backend API for activation (activate, status)
- ✅ Create team locked modal for unactivated coaches
- ✅ Locked upload modal for players without activated teams

### Phase 4: Match Management
- ✅ Upload modal with Playsight link input
- ✅ Player identification interface (multi-frame click-to-identify)
- ✅ Processing status component with real-time updates
- ✅ Match detail page with court diagram placeholder
- ✅ Match stats display component
- ✅ Coach can upload matches for team members
- ✅ Player filter dropdown for coaches on dashboard
- ✅ Match cards with date grouping

### Phase 5: Backend API
- ✅ FastAPI endpoints for teams (create, join, list, members)
- ✅ FastAPI endpoints for matches (list, get, create)
- ✅ FastAPI endpoints for videos (upload, identify player, status)
- ✅ FastAPI endpoints for stats (player stats, season stats)
- ✅ Authentication middleware with Supabase token verification
- ✅ Role-based access control in API endpoints
- ✅ Data isolation for coaches (only see their team's matches)

### Phase 6: Database & Security
- ✅ Complete database schema with all tables
- ✅ Row Level Security (RLS) policies configured
- ✅ Database triggers for automatic user profile creation
- ✅ Team membership junction table
- ✅ Match ownership tracking
- ✅ Data isolation between coaches and teams

---

## 🚧 Still Needs Implementation

### High Priority

1. **Playsight Frame Extraction**
   - Backend needs to extract frames from Playsight videos for player identification
   - Currently using placeholder frames
   - Need to research Playsight API or scraping methods

2. **CV Backend Integration**
   - Connect to actual CV processing pipeline (`old/src/core/tennis_CV.py`)
   - Implement video processing workflow
   - Parse JSON output from CV backend
   - Store processed data in database

3. **Player Tracking**
   - Implement color recognition based on player identification clicks
   - Track player position throughout video
   - Generate shot data from tracking

4. **Court Visualization with Real Data**
   - Parse shot data from JSON output
   - Render shots on court diagram with correct positions
   - Implement clickable shot lines with video timestamp navigation
   - Color code shots (winners/errors/in-play)

5. **Statistics Visualization**
   - Add Recharts components for detailed statistics
   - Create charts for winners/errors over time
   - Shot distribution charts
   - Performance by match charts
   - Shot pattern visualizations

### Medium Priority

6. **Video Processing Workflow**
   - Complete async video processing pipeline
   - Queue system for video processing
   - Error handling and retry logic
   - Progress tracking and updates

7. **Error Handling & User Feedback**
   - Comprehensive error handling throughout app
   - User-friendly error messages
   - Loading states and indicators
   - Success notifications

8. **Testing**
   - Unit tests for components
   - Integration tests for API endpoints
   - End-to-end testing for user flows
   - Database migration tests

### Low Priority

9. **Additional Features**
   - Match sharing functionality
   - Export statistics to PDF/CSV
   - Advanced filtering and search
   - Match comparison tools
   - Player performance trends

---

## 📋 Setup Instructions

### Prerequisites

- Node.js 18+ and npm
- Python 3.8+
- Supabase account (free tier works)

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
2. Copy the following:
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
   - You should see: **"Success. No rows returned"** (this is normal)

3. **Verify Tables Created**
   - Go to **Table Editor** in the left sidebar
   - You should see these tables:
     - ✅ `users` - User profiles (extends auth.users, includes activation_key and activated_at)
     - ✅ `teams` - Teams with unique codes
     - ✅ `team_members` - Junction table for team membership
     - ✅ `matches` - Match records
     - ✅ `match_data` - JSON data from CV processing
     - ✅ `shots` - Individual shot records
     - ✅ `player_identifications` - Player identification data

### Step 3.5: Understanding Activation Keys

**How Activation Works:**
- Coaches need activation keys to create teams and upload matches
- Players need to be on a team with an activated coach to upload matches
- Activation keys are **team-based** - one key activates the entire team
- When someone joins a team with an activated coach, they automatically get the same activation key

**To Add Activation Keys Manually (for testing):**

1. **Add Key to Purchasing Coach:**
   ```sql
   -- In Supabase SQL Editor
   UPDATE public.users 
   SET activation_key = 'ABC123'
   WHERE email = 'coach-email@example.com' 
     AND role = 'coach';
   ```

2. **Check Activation Status:**
   ```sql
   SELECT 
     email, 
     name, 
     role, 
     activation_key, 
     activated_at,
     CASE 
       WHEN activated_at IS NOT NULL THEN 'Activated'
       WHEN activation_key IS NOT NULL THEN 'Key Assigned (Not Activated)'
       ELSE 'No Key Assigned'
     END as status
   FROM public.users 
   WHERE role = 'coach'
   ORDER BY created_at DESC;
   ```

**Activation Flow:**
1. Admin adds `activation_key` to coach's user record (via SQL)
2. Coach enters key in UI → Account activates
3. Other coaches/players join team → Automatically get same key and activate

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

### Step 6: Configure Email Authentication

For easier development, disable email confirmation:

1. Go to **Authentication** → **Settings** in Supabase dashboard
2. Under "Email Auth", toggle off **"Enable email confirmations"**
3. (Re-enable for production!)

This lets you sign up and immediately sign in without checking email.

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

**Terminal 2 - Backend:**
```bash
cd backend
source ../tennis_env/bin/activate  # On Windows: ..\tennis_env\Scripts\activate
uvicorn main:app --reload --port 8000
```

### Step 8: Test the Setup

1. **Visit the App**
   - Go to http://localhost:3000
   - You should see the landing page

2. **Create a Test Account**
   - Click "Sign Up" in the top right
   - Select role (Coach or Player)
   - Enter: Name, Email, Password
   - Click "Sign up"
   - You should be redirected to `/dashboard`

3. **Test Team Creation (Coach)**
   - Go to `/teams`
   - Click "Create New Team"
   - Enter team name
   - See team code generated

4. **Test Team Joining (Player)**
   - Sign up as a player
   - Go to `/teams`
   - Enter team code from coach
   - Click "Join"

---

## 📁 Project Structure

```
tennis_analytics/
├── frontend/                    # Next.js frontend
│   ├── app/                     # App Router pages
│   │   ├── page.tsx            # Landing page
│   │   ├── dashboard/          # Dashboard page
│   │   ├── stats/             # Stats page
│   │   ├── teams/             # Teams page
│   │   ├── profile/            # Profile page
│   │   ├── matches/[id]/      # Match detail page
│   │   └── layout.tsx         # Root layout
│   ├── components/             # React components
│   │   ├── auth/              # Authentication components
│   │   ├── landing/           # Landing page components
│   │   ├── layout/            # Layout components (sidebar, etc.)
│   │   ├── team/              # Team management components
│   │   ├── upload/            # Upload and processing components
│   │   ├── court/             # Court visualization components
│   │   ├── match/              # Match-related components
│   │   └── ui/                # shadcn/ui components
│   ├── hooks/                  # Custom React hooks
│   │   ├── useAuth.ts         # Authentication hook
│   │   ├── useMatches.ts      # Matches data hook
│   │   ├── useTeams.ts        # Teams data hook
│   │   ├── useProfile.ts      # User profile hook
│   │   └── useTeamMembers.ts  # Team members hook
│   ├── lib/                    # Utilities & configuration
│   │   ├── supabase/          # Supabase client setup
│   │   ├── utils.ts           # Utility functions
│   │   └── types.ts           # TypeScript types
│   └── .env.local             # Environment variables (not in git)
├── backend/                    # FastAPI backend
│   ├── api/                    # API route handlers
│   │   ├── teams.py           # Team endpoints
│   │   ├── matches.py         # Match endpoints
│   │   ├── videos.py          # Video endpoints
│   │   └── stats.py           # Stats endpoints
│   ├── services/               # Business logic
│   │   ├── cv_integration.py  # CV backend integration
│   │   ├── player_tracker.py  # Player tracking service
│   │   └── playsight.py       # Playsight integration
│   ├── auth.py                 # Authentication middleware
│   ├── main.py                 # FastAPI app entry point
│   └── requirements.txt        # Python dependencies
├── supabase/                   # Database schema
│   └── schema.sql             # Complete database schema
├── old/                        # Legacy CV backend code
├── SAM3/                       # SAM3 model files (large, not in git)
├── start_frontend.sh          # Frontend startup script
├── start_backend.sh           # Backend startup script
└── README.md                  # This file
```

---

## 🔄 User Flows

### Authentication Flow

1. **Sign Up (Coach)**
   - Visit landing page
   - Click "Sign Up" → Modal opens
   - Select "Coach" role
   - Fill in: Name, Email, Password
   - Click "Sign up"
   - Auto-redirect to `/dashboard`
   - Role saved to database via trigger

2. **Sign Up (Player)**
   - Same as coach, but select "Player" role

3. **Sign In**
   - Click "Sign In" → Modal opens
   - Enter email and password
   - Click "Sign in"
   - Redirect to `/dashboard`

### Activation Flow

1. **Coach Purchases & Gets Activation Key**
   - Admin manually adds activation key to coach's user record in database
   - Coach signs up or logs in
   - Coach navigates to Teams page
   - Coach sees activation key input at top
   - Coach enters activation key
   - Account activates → Can now create teams and upload matches

2. **Coach Joins Activated Team (Alternative)**
   - Coach signs up (no activation key needed initially)
   - Coach navigates to Teams page (accessible even without activation)
   - Coach enters team code from another activated coach
   - Coach automatically gets same activation key and is activated
   - Can now use all features

3. **Player Joins Activated Team**
   - Player signs up
   - Player navigates to Teams page
   - Player enters team code
   - If team has activated coach → Player automatically gets activation key and is activated
   - Player can now upload matches

### Team Management Flow

1. **Coach Creates Team** (Requires Activation)
   - Coach must be activated first (via activation key or joining activated team)
   - Go to `/teams`
   - Click "Create New Team"
   - Enter team name
   - See team code displayed
   - Team appears in "Your Teams" list
   - If not activated → Modal explains need for activation

2. **Player Joins Team**
   - Go to `/teams`
   - Enter team code from coach
   - Click "Join"
   - If team has activated coach → Auto-activated with shared key
   - Team appears in "Your Teams" list

3. **Coach Joins Existing Team**
   - Go to `/teams` (accessible even without activation)
   - Enter team code from another coach
   - Click "Join"
   - If team has activated coach → Auto-activated with shared key
   - Coach joins as coach role
   - Team appears in "Your Teams" list

### Match Upload Flow

1. **Player Uploads Own Match** (Requires Activated Team)
   - Player must be on a team with an activated coach
   - Click floating "+" button (locked if no activated team)
   - If locked → Modal explains need to join activated team
   - If unlocked → Modal opens
   - Enter Playsight link
   - Click "Upload"
   - Redirect to `/matches/[id]/identify`
   - Click on yourself in frames
   - Submit identification
   - Match appears in dashboard

2. **Coach Uploads Match for Player** (Requires Activation)
   - Coach must be activated
   - Click floating "+" button (locked if not activated)
   - If unlocked → Modal opens
   - **Select player** from dropdown (team members)
   - Enter Playsight link
   - Click "Upload"
   - Redirect to identification page
   - Coach clicks on player in frames
   - Submit identification
   - **Match appears in BOTH coach's and player's dashboard**

### Dashboard Flow

1. **Player View**
   - See all own matches
   - Grouped by date
   - Click match to view details

2. **Coach View**
   - See all team member matches
   - Player filter dropdown at top
   - Default: "All Players"
   - Can filter by specific player
   - Matches grouped by date

### Match Detail Flow

1. **View Match (Processing)**
   - Click match card
   - See processing status
   - No court diagram (status !== 'completed')
   - Message: "Processing in progress..."

2. **View Match (Completed)**
   - Click match card
   - See court diagram with shots
   - Click shot line → Video panel opens
   - Video jumps to shot timestamp
   - See match stats below court

---

## 🗄️ Database Schema

### Tables

- **`users`** - User profiles (extends auth.users)
  - `id` (UUID, primary key)
  - `email` (TEXT, unique)
  - `name` (TEXT)
  - `role` (TEXT: 'coach' or 'player')
  - `activation_key` (TEXT, unique) - Activation key for coaches/players
  - `activated_at` (TIMESTAMP) - When account was activated
  - `team_id` (UUID, nullable, references teams)

- **`teams`** - Teams with unique codes
  - `id` (UUID, primary key)
  - `name` (TEXT)
  - `code` (TEXT, unique, 6 characters)
  - `created_at`, `updated_at`

- **`team_members`** - Team membership junction table
  - `id` (UUID, primary key)
  - `team_id` (UUID, references teams)
  - `user_id` (UUID, references users)
  - `role` (TEXT: 'coach' or 'player')
  - `joined_at` (TIMESTAMP)

- **`matches`** - Match records
  - `id` (UUID, primary key)
  - `user_id` (UUID, references users - the player)
  - `player_name` (TEXT, nullable)
  - `playsight_link` (TEXT)
  - `video_url` (TEXT, nullable)
  - `status` (TEXT: 'pending', 'processing', 'completed', 'failed')
  - `processed_at` (TIMESTAMP, nullable)
  - `created_at`, `updated_at`

- **`match_data`** - JSON data from CV processing
  - `id` (UUID, primary key)
  - `match_id` (UUID, references matches)
  - `json_data` (JSONB)
  - `stats_summary` (JSONB, nullable)
  - `created_at`, `updated_at`

- **`shots`** - Individual shot records
  - `id` (UUID, primary key)
  - `match_id` (UUID, references matches)
  - `shot_type` (TEXT, nullable)
  - `start_pos` (JSONB: {x, y})
  - `end_pos` (JSONB: {x, y})
  - `timestamp` (INTEGER - frame number)
  - `video_timestamp` (REAL - seconds)
  - `result` (TEXT: 'winner', 'error', 'in_play')
  - `created_at`

- **`player_identifications`** - Player identification data
  - `id` (UUID, primary key)
  - `match_id` (UUID, references matches)
  - `frame_data` (JSONB)
  - `selected_player_coords` (JSONB: {x, y})
  - `created_at`

### Row Level Security (RLS)

All tables have RLS enabled with policies that:
- Users can view/update their own data
- Coaches can view all team member matches
- Team members can view each other's basic info
- Data is isolated between different coaches' teams

### Triggers

- **`handle_new_user()`** - Automatically creates user profile when auth user is created
  - Reads role from `auth.users.raw_user_meta_data->>'role'`
  - Defaults to 'player' if not specified
  - Validates role is 'coach' or 'player'

- **`update_updated_at_column()`** - Automatically updates `updated_at` timestamp on row updates

---

## 🔌 API Endpoints

### Teams

- `GET /api/teams/my-teams` - Get user's teams
- `POST /api/teams/create` - Create team (coaches only)
- `POST /api/teams/join` - Join team using code (coaches and players)
- `GET /api/teams/{team_id}/members` - Get team members

### Matches

- `GET /api/matches` - Get user's matches (coaches see all team matches)
- `POST /api/matches` - Create match (with optional `user_id` for coach uploads)
- `GET /api/matches/{id}` - Get match details

### Videos

- `POST /api/videos/upload` - Upload video (creates match)
- `POST /api/videos/identify-player` - Submit player identification
- `GET /api/videos/{match_id}/status` - Get processing status

### Activation

- **`POST /api/activation/activate`** - Activate coach account with key
  - Body: `{ "activation_key": "ABC123" }`
  - Returns: `{ "message": "Account activated successfully", "activated": true }`
  - Only coaches can activate
  - Key must match the user's record in database

- **`GET /api/activation/status`** - Get activation status
  - Returns: `{ "is_activated": true/false, "role": "coach"/"player", "activated_at": timestamp }`

### Stats

- `GET /api/stats/player/{player_id}` - Get player stats
- `GET /api/stats/season` - Get season stats for user

### Health

- `GET /api/health` - Health check endpoint

All endpoints require authentication via Bearer token (Supabase JWT).

---

## 🧪 Testing

### Quick Test Checklist

1. **Authentication**
   - ✅ Sign up as coach
   - ✅ Sign up as player
   - ✅ Sign in with both accounts
   - ✅ Verify role is saved correctly

2. **Activation System**
   - ✅ Add activation key to coach via SQL
   - ✅ Coach enters key and activates account
   - ✅ Coach can create teams after activation
   - ✅ Coach can join activated team (auto-activation)
   - ✅ Player joins activated team (auto-activation)
   - ✅ Upload button locked for unactivated coaches
   - ✅ Upload button locked for players without activated teams

3. **Team Management**
   - ✅ Coach creates team (requires activation)
   - ✅ Player joins team
   - ✅ Coach joins existing team
   - ✅ View team members (coaches and players)

4. **Match Upload**
   - ✅ Player uploads own match (requires activated team)
   - ✅ Coach uploads match for player (requires activation)
   - ✅ Match appears in both dashboards
   - ✅ Upload button shows lock icon when locked

5. **Dashboard**
   - ✅ Player sees own matches
   - ✅ Coach sees all team matches
   - ✅ Coach can filter by player
   - ✅ Activation key input shown for unactivated coaches

6. **Navigation**
   - ✅ All pages accessible
   - ✅ Teams page accessible even for unactivated coaches
   - ✅ Profile page shows correct info
   - ✅ Sign out works

### API Testing

Visit http://localhost:8000/docs for interactive API documentation (Swagger UI).

---

## 🐛 Troubleshooting

### Frontend Issues

**Frontend won't start**
- Check Node.js 18+ is installed: `node --version`
- Verify `frontend/.env.local` has valid Supabase credentials
- Try: `cd frontend && npm install` again
- Check port 3000 is not in use: `lsof -i :3000`

**Landing page redirects to /login**
- Check middleware.ts doesn't have redirect logic
- Verify `app/page.tsx` is the landing page component
- Clear `.next` cache: `rm -rf frontend/.next`

**Buttons don't show pointer cursor**
- Check `globals.css` has cursor rules
- Verify button components have `cursor-pointer` class

### Backend Issues

**Backend won't start**
- Check Python 3.8+ is installed: `python3 --version`
- Activate virtual environment: `source tennis_env/bin/activate`
- Install dependencies: `pip install -r backend/requirements.txt`
- Check `backend/.env` has valid Supabase credentials
- Check port 8000 is not in use: `lsof -i :8000`

**ModuleNotFoundError**
- Activate virtual environment
- Run `pip install -r backend/requirements.txt`

### Database Issues

**Database errors**
- Make sure you've run `supabase/schema.sql` in Supabase SQL Editor
- Check all tables are created in Table Editor
- Verify Supabase credentials in `.env` files are correct

**User profile not created**
- Check database trigger was created
- In Supabase SQL Editor, run: `SELECT * FROM public.users;`
- If empty, re-run the schema.sql file

**Role not saved correctly**
- Check trigger function `handle_new_user()` is updated
- Verify role is passed in signup metadata
- Check browser console for "Signing up with role: coach/player" log

**Activation key not working**
- Verify activation key was added to user record: `SELECT activation_key FROM public.users WHERE email = 'coach@example.com';`
- Check key matches exactly (case-insensitive, but stored as entered)
- Verify `activated_at` is NULL before activation
- Check backend logs for activation errors
- Ensure coach is entering key in Teams page activation input

**Can't create team / upload matches**
- For coaches: Check if account is activated (`activated_at IS NOT NULL`)
- For players: Check if team has activated coach (any coach on team with `activated_at IS NOT NULL`)
- Check activation status via API: `GET /api/activation/status`

### Authentication Issues

**Can't sign up / Email not sending**
- Check **Authentication** → **Settings** → **Email Auth** in Supabase
- For development, disable email confirmation
- Check Supabase project is active (not paused)

**"relation auth.users does not exist" warning**
- This is normal! `auth.users` is created automatically by Supabase
- You can safely ignore this warning

### CORS Issues

**CORS errors in browser**
- Verify `ALLOWED_ORIGINS=http://localhost:3000` in `backend/.env`
- Check `backend/main.py` has CORS middleware configured
- Restart backend server after changing `.env`

---

## 🔒 Data Isolation

### How It Works

- **Players belong to teams** (via `team_members` table)
- **Coaches create teams** and players join via team codes
- **Each team is isolated** - coaches only see data from players on their teams
- **Multiple coaches are isolated** - Coach A cannot see Coach B's team data

### Example

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
- ✅ Players only see their own matches

### Enforcement

- **RLS Policies**: Database-level security ensures coaches can only query matches from their team members
- **Backend API**: Additional filtering ensures only team member matches are returned
- **Team Membership**: Players must join teams via codes, ensuring proper team assignment

---

## 📝 Development Notes

### Frontend Development

- Uses Next.js 14+ App Router
- Pages in `app/` directory
- Components in `components/` directory
- Hooks in `hooks/` directory
- Supabase client in `lib/supabase/`
- All API calls go through backend (not direct Supabase)

### Backend Development

- FastAPI application in `backend/main.py`
- Add API routes in `backend/api/`
- Add services in `backend/services/`
- Add models in `backend/models/`
- All database access goes through Supabase client

### Database Changes

- Schema defined in `supabase/schema.sql`
- Run SQL directly in Supabase SQL Editor
- Row Level Security (RLS) policies are configured
- Use `CREATE OR REPLACE FUNCTION` for function updates

### Code Style

- Frontend: TypeScript with strict mode
- Backend: Python with type hints
- Follow existing patterns in codebase
- Use shadcn/ui components for UI consistency

---

## 📦 Large Files & Git

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
- `tennis_env/` - Python virtual environment
- `frontend/node_modules/` - Node dependencies

**Media & Data:**
- Video files (`.mp4`, `.avi`, `.mov`, etc.)
- Image files (`.jpg`, `.png`, etc.)

**Build Outputs:**
- `frontend/.next/` - Next.js build output
- `frontend/out/` - Next.js export output

**Environment Files:**
- `.env`, `.env.local` - Sensitive credentials (never committed)

---

## 🚀 Deployment (Planned)

### Frontend (Vercel)
- Connect GitHub repository
- Set environment variables in Vercel dashboard
- Automatic deployments on push to main

### Backend (Railway)
- Connect GitHub repository
- Set environment variables in Railway dashboard
- Automatic deployments on push to main

### Database (Supabase)
- Already hosted on Supabase
- No additional deployment needed
- Update RLS policies as needed

---

## 📚 Additional Resources

- [Next.js Documentation](https://nextjs.org/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Supabase Documentation](https://supabase.com/docs)
- [TanStack Query Documentation](https://tanstack.com/query/latest)
- [shadcn/ui Components](https://ui.shadcn.com/)

---

## 🤝 Contributing

This is a private project. For questions or issues, contact the project maintainer.

---

## 📄 License

Private project - All rights reserved.

---

**Last Updated**: December 2024  
**Version**: 1.0.0  
**Status**: Active Development
