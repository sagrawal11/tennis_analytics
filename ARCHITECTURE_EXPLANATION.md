# Courtvision Architecture - Complete Backend Explanation

## 🏗️ High-Level Architecture

Yes, you're **absolutely correct**! The frontend and backend run on **separate servers**:

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│   Frontend      │         │    Backend      │         │   Database      │
│   (Vercel)      │◄───────►│   (Railway)     │◄───────►│   (Supabase)    │
│                 │  HTTP   │                 │  SQL    │                 │
│  - Next.js      │  API    │  - FastAPI      │         │  - PostgreSQL   │
│  - React UI     │  Calls  │  - CV Models   │         │  - Auth         │
│  - Lightweight  │         │  - Heavy ML     │         │  - Realtime     │
└─────────────────┘         └─────────────────┘         └─────────────────┘
```

---

## 📱 Frontend Server (Lightweight)

**Deployment:** Vercel (or similar static hosting)

**What it does:**
- Serves the React/Next.js UI to users' browsers
- Handles routing, page rendering, and user interactions
- Makes HTTP API calls to the backend
- **No heavy processing** - just displays data and sends requests

**Resource Requirements:**
- **Very light** - can run on basic hosting
- Static files (HTML, CSS, JavaScript)
- Minimal CPU/memory needed
- Fast global CDN distribution

**Key Files:**
- `frontend/app/` - Pages (dashboard, teams, stats, etc.)
- `frontend/components/` - UI components
- `frontend/hooks/` - API calls to backend

---

## ⚙️ Backend Server (Heavy - CV/ML Processing)

**Deployment:** Railway, AWS EC2, Google Cloud, or similar (needs GPU for ML models)

**What it does:**
1. **API Server** - Handles HTTP requests from frontend
2. **Computer Vision Processing** - Runs ML models to analyze tennis videos
3. **Database Operations** - Reads/writes to Supabase
4. **Video Processing** - Downloads, processes, and analyzes videos

**Resource Requirements:**
- **Heavy** - needs significant resources:
  - **GPU** (recommended) - for SAM-3d-body, SAM3, YOLO models
  - **CPU** - for video processing, frame extraction
  - **RAM** - 8GB+ (models load into memory)
  - **Storage** - for temporary video files during processing
  - **Network** - to download Playsight videos

**Key Components:**

### 1. FastAPI Server (`backend/main.py`)
- Receives HTTP requests from frontend
- Handles authentication (JWT tokens from Supabase)
- Routes requests to appropriate endpoints
- Returns JSON responses

### 2. API Endpoints (`backend/api/`)
- **Teams** (`teams.py`) - Create/join teams, manage members
- **Matches** (`matches.py`) - Create matches, list matches
- **Videos** (`videos.py`) - Upload videos, identify players, check status
- **Stats** (`stats.py`) - Get player/team statistics
- **Activation** (`activation.py`) - Handle activation keys

### 3. Services (`backend/services/`)
- **`cv_integration.py`** - Orchestrates CV processing
- **`player_tracker.py`** - Tracks players using color recognition
- **`playsight.py`** - Extracts frames from Playsight videos

### 4. Computer Vision Models (in project root)
- **SAM-3d-body/** - Player pose estimation (3D mesh)
- **SAM3/** - Ball detection
- **YOLO models** - Human/ball detection
- **TrackNet, RF-DETR** - Additional ball tracking
- **Court detection** - Tennis court line detection

---

## 🗄️ Database (Supabase)

**What it is:** PostgreSQL database hosted by Supabase

**What it stores:**
- User accounts and authentication
- Teams and team members
- Matches and match data
- Player identifications
- Shot data and statistics
- Activation keys

**Why separate:**
- Managed service (no server to maintain)
- Built-in authentication
- Real-time subscriptions (for processing status updates)
- Row Level Security (data isolation)

---

## 🔄 Complete User Flow: Upload → Processing → Results

Let's trace what happens when a user uploads a video:

### Step 1: User Uploads Video (Frontend → Backend → Database)

```
User clicks "Upload" button
    ↓
Frontend opens UploadModal
    ↓
User enters Playsight link, date, opponent, notes
    ↓
Frontend calls: POST /api/matches
    ↓
Backend creates match record in Supabase
    ↓
Status: "pending"
    ↓
Returns match_id to frontend
```

**Code:** `frontend/components/upload/upload-modal.tsx` → `backend/api/matches.py`

---

### Step 2: Player Identification (Frontend → Backend → Database)

```
Frontend extracts frames from Playsight video
    ↓
Shows user 3-5 frames
    ↓
User clicks on themselves in each frame
    ↓
Frontend calls: POST /api/videos/identify-player
    ↓
Backend stores player coordinates in database
    ↓
Status: "processing"
    ↓
Triggers CV processing (async)
```

**Code:** `frontend/components/upload/upload-modal.tsx` → `backend/api/videos.py` → `backend/services/cv_integration.py`

---

### Step 3: Computer Vision Processing (Backend - Heavy Work)

```
Backend receives processing trigger
    ↓
Downloads video from Playsight (or uses cached)
    ↓
Loads ML models into memory:
    - SAM-3d-body (player tracking)
    - SAM3 (ball detection)
    - YOLO (human/ball detection)
    - Court detector
    ↓
Processes video frame-by-frame:
    - Detects players
    - Tracks ball
    - Detects bounces
    - Classifies shots
    - Maps to court coordinates
    ↓
Generates JSON output with:
    - All shots (positions, timestamps, results)
    - Player positions
    - Ball trajectory
    - Statistics
    ↓
Stores results in Supabase match_data table
    ↓
Status: "completed"
```

**Code:** `backend/services/cv_integration.py` → calls `old/src/core/tennis_CV.py` or similar

**Time:** Can take 30 minutes to 2+ hours depending on video length and processing options

---

### Step 4: View Results (Frontend → Backend → Database → Frontend)

```
User navigates to match detail page
    ↓
Frontend calls: GET /api/matches/{id}
    ↓
Backend fetches match + match_data from Supabase
    ↓
Returns JSON with all shot data
    ↓
Frontend renders:
    - Interactive court diagram
    - Clickable shots (green=winner, red=error, blue=in-play)
    - Video player with timeline
    - Statistics
```

**Code:** `frontend/app/matches/[id]/page.tsx` → `backend/api/matches.py`

---

## 🔌 How Frontend and Backend Communicate

### Authentication Flow

```
1. User signs in on frontend
   ↓
2. Supabase Auth returns JWT token
   ↓
3. Frontend stores token
   ↓
4. Every API call includes token in header:
   Authorization: Bearer <jwt_token>
   ↓
5. Backend validates token with Supabase
   ↓
6. Backend extracts user_id from token
   ↓
7. Backend uses user_id for database queries
```

### API Communication

**Frontend makes HTTP requests:**
```typescript
// Example: frontend/hooks/useTeams.ts
const response = await fetch('http://backend-url/api/teams/my-teams', {
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  }
})
```

**Backend responds with JSON:**
```python
# Example: backend/api/teams.py
@router.get("/my-teams")
async def get_my_teams(user_id: str = Depends(get_user_id)):
    # Query Supabase
    teams = supabase.table("teams").select("*").eq("user_id", user_id).execute()
    return {"teams": teams.data}
```

---

## 🚀 Deployment Strategy

### Frontend (Vercel)
- **Why Vercel:** Optimized for Next.js, automatic deployments, global CDN
- **Setup:** Connect GitHub repo, Vercel auto-detects Next.js
- **Environment Variables:** 
  - `NEXT_PUBLIC_SUPABASE_URL`
  - `NEXT_PUBLIC_SUPABASE_ANON_KEY`
  - `NEXT_PUBLIC_BACKEND_URL` (your backend server URL)

### Backend (Railway/AWS/GCP)
- **Why Railway:** Easy Python deployment, supports GPU instances
- **Alternative:** AWS EC2 (with GPU), Google Cloud Run, DigitalOcean
- **Setup:**
  1. Install Python dependencies (`pip install -r requirements.txt`)
  2. Install CV models (SAM-3d-body, SAM3, etc.)
  3. Set environment variables
  4. Run `uvicorn main:app --host 0.0.0.0 --port 8000`
- **Environment Variables:**
  - `SUPABASE_URL`
  - `SUPABASE_SERVICE_ROLE_KEY`
  - `ALLOWED_ORIGINS` (frontend URL for CORS)
  - `PLAYSIGHT_API_KEY` (if needed)

### Database (Supabase)
- **Already hosted** - no deployment needed
- Just need to run `schema.sql` in Supabase SQL Editor

---

## 💾 Resource Requirements Summary

### Frontend Server (Vercel)
- **CPU:** Minimal (just serves static files)
- **RAM:** < 1GB
- **Storage:** Minimal (just code)
- **Cost:** Free tier usually sufficient

### Backend Server (Railway/AWS)
- **CPU:** 4+ cores recommended
- **RAM:** 8GB+ (models load into memory)
- **GPU:** Recommended (NVIDIA GPU for faster ML inference)
- **Storage:** 50GB+ (for temporary video files)
- **Network:** Good bandwidth (to download videos)
- **Cost:** $20-100+/month depending on instance

### Database (Supabase)
- **Managed service** - no server management
- **Free tier:** 500MB database, 2GB bandwidth
- **Paid tier:** Scales automatically

---

## 🔄 Async Processing (Future Enhancement)

Currently, video processing is **synchronous** (blocks the API request). In production, you'd want:

**Task Queue System (Celery + Redis/RabbitMQ):**
```
1. User uploads video
2. Backend immediately returns: "Processing started"
3. Backend adds task to queue
4. Worker process picks up task
5. Worker processes video (can take hours)
6. Worker updates database when done
7. Frontend polls status or uses WebSocket for updates
```

This prevents:
- API timeouts (processing takes too long)
- Blocking other requests
- User waiting for response

---

## 📁 Key Files Reference

### Frontend
- `frontend/app/` - Pages
- `frontend/components/` - UI components
- `frontend/hooks/` - API calls (useTeams, useMatches, etc.)

### Backend
- `backend/main.py` - FastAPI app entry point
- `backend/api/` - API endpoints
- `backend/services/` - Business logic
- `backend/auth.py` - Authentication helpers

### CV/ML Models
- `SAM-3d-body/` - Player tracking
- `SAM3/` - Ball detection
- `old/src/core/tennis_CV.py` - Main CV processing pipeline
- `hero-video/` - Promotional video processing

### Database
- `supabase/schema.sql` - Database schema

---

## 🎯 Summary

**You're correct:**
- ✅ Frontend and backend are on **separate servers**
- ✅ Frontend is **lightweight** (just UI)
- ✅ Backend is **heavy** (CV/ML processing)
- ✅ Database is **separate** (Supabase)

**The flow:**
1. User interacts with frontend (Vercel)
2. Frontend calls backend API (Railway)
3. Backend processes with CV models (heavy computation)
4. Backend stores results in Supabase
5. Frontend displays results to user

**Next steps for you:**
1. Deploy frontend to Vercel (easy, automatic)
2. Deploy backend to Railway (needs GPU for ML)
3. Connect them via environment variables
4. Test the full flow end-to-end
