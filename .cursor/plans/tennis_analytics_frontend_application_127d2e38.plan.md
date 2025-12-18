---
name: Tennis Analytics Frontend Application
overview: Build a full-stack tennis analytics web application with user authentication, video processing workflow, interactive court visualization, and comprehensive stats tracking for coaches and players.
todos:
  - id: setup_project_structure
    content: Create frontend and backend directory structures, initialize Next.js and FastAPI projects
    status: completed
  - id: setup_supabase
    content: Create Supabase project, set up database schema (users, teams, matches, shots tables), configure RLS policies
    status: completed
  - id: setup_supabase_auth
    content: Configure Supabase Auth with email/password (simplified from OAuth), set up Supabase client in Next.js, create login/signup pages
    status: completed
  - id: setup_frontend
    content: Initialize Next.js 14+ project with Tailwind CSS, install shadcn/ui, set up project structure
    status: completed
  - id: build_layout
    content: Create main layout with left sidebar navigation, floating action button, and responsive design using Tailwind + shadcn/ui
    status: completed
  - id: build_dashboard
    content: Build dashboard page with date-organized match listings, match cards, and player name display
    status: completed
  - id: build_stats_page
    content: Create stats page with coach/player views, season stats, and per-game breakdowns using Recharts
    status: completed
  - id: build_court_diagram
    content: Create interactive tennis court diagram component with SVG/canvas, court markings, and shot rendering
    status: completed
  - id: implement_shot_visualization
    content: Parse JSON from CV backend, render clickable shot lines with color coding (errors/winners), hover states
    status: completed
  - id: build_video_panel
    content: Create side panel component for Playsight video embedding with timestamp navigation
    status: completed
  - id: build_upload_modal
    content: Create upload modal with Playsight link input, validation, and submission
    status: completed
  - id: implement_player_identification
    content: Build player identification interface with multiple frame display and click-to-identify functionality
    status: completed
  - id: implement_processing_status
    content: Create real-time processing status display using Supabase Realtime subscriptions with progress updates and completion notifications
    status: completed
  - id: implement_team_management
    content: Build team code system for coaches to create teams and players to join via code
    status: completed
  - id: build_backend_api
    content: Create FastAPI endpoints for matches, stats, videos, and teams (auth handled by Supabase)
    status: completed
  - id: integrate_cv_backend
    content: Integrate existing tennis_analytics CV backend, create processing pipeline, JSON output parsing
    status: completed
  - id: implement_player_tracking
    content: Implement color recognition and player tracking based on identification clicks
    status: completed
  - id: research_playsight
    content: Research Playsight API or scraping methods for video extraction and embedding
    status: completed
---

# Tennis Analytics Frontend Application - Implementation Plan

## System Overview

A web application for tennis analytics with two user types (coaches and players), video processing integration, interactive court visualization, and detailed statistics tracking.

## Architecture

### User Flow

```
Login (OAuth) → Dashboard → Match Detail → Stats Page
                ↓
            Upload Video → Player Identification → Processing → Results
```

### Key Components

1. **Frontend**: Next.js 14+ web application with interactive court visualization
2. **Backend API**: FastAPI server handling data and CV integration
3. **Database & Auth**: Supabase (PostgreSQL + Auth + Realtime)
4. **CV Processing**: Integration with existing tennis_analytics backend
5. **Video Storage**: Playsight-hosted videos (embedded)

## Database Schema

### Core Tables

- `users` (id, email, name, role, team_id, created_at)
- `teams` (id, name, code, created_at)
- `team_members` (team_id, user_id, role, joined_at)
- `matches` (id, user_id, playsight_link, video_url, status, processed_at, created_at)
- `match_data` (id, match_id, json_data, stats_summary)
- `shots` (id, match_id, shot_type, start_pos, end_pos, timestamp, video_timestamp, result)
- `player_identifications` (id, match_id, frame_data, selected_player_coords)

## Implementation Phases

### Phase 1: Project Setup & Foundation

**1.1 Project Structure**

- Create frontend directory structure (React/Next.js recommended)
- Set up FastAPI backend structure
- Configure PostgreSQL database
- Set up development environment (Docker optional)

**1.2 Authentication System**

- Implement OAuth (Google/Apple) authentication
- User session management
- Role-based access control (coach vs player)
- Protected routes

**1.3 Database Setup**

- Create PostgreSQL schema
- Set up database migrations (Alembic)
- Seed initial data structure

**Files to Create:**

- `frontend/` directory with React/Next.js setup
- `backend/` directory with FastAPI structure
- `backend/database/` with models and migrations
- `backend/auth/` with OAuth handlers

### Phase 2: Core UI Components

**2.1 Layout & Navigation**

- Left sidebar component (Dashboard, Stats, Logout)
- Main content area
- Responsive layout
- Bottom-right circular plus button (floating action button)

**2.2 Dashboard Page**

- Game listing component
- Date-based dropdown organization (coaches)
- Player name display per match
- Match card components
- Loading states

**2.3 Stats Page**

- Coach view: Player selection interface
- Player view: Personal stats display
- Season stats breakdown
- Per-game stats visualization
- Stats cards/charts

**Files to Create:**

- `frontend/components/layout/` (Sidebar, MainLayout)
- `frontend/app/dashboard/page.tsx` (Next.js App Router)
- `frontend/app/stats/page.tsx`
- `frontend/components/match/MatchCard.tsx`
- `frontend/components/stats/StatsDisplay.tsx`
- `frontend/hooks/useMatches.ts` (TanStack Query hook)

### Phase 3: Match Detail & Court Visualization

**3.1 Court Diagram Component**

- Top-down tennis court SVG/canvas component
- Court markings (baseline, service boxes, net, etc.)
- Responsive sizing

**3.2 Shot Visualization**

- Parse JSON from CV backend
- Render shot trajectories (lines connecting start/end points)
- Color coding: errors (red), winners (green), in-play (blue)
- Clickable shot lines
- Hover states

**3.3 Video Integration**

- Side panel component for video playback
- Playsight video embedding
- Timestamp navigation (jump to specific moment)
- Video controls

**3.4 Stats Breakdown Section**

- Errors/winners per game, set, match
- Detailed stats display (first serve %, unforced errors, etc.)
- Stats cards/graphs below court diagram

**Files to Create:**

- `frontend/components/Court/CourtDiagram.tsx`
- `frontend/components/Court/ShotLine.tsx`
- `frontend/components/Video/VideoPanel.tsx`
- `frontend/components/Stats/MatchStats.tsx`
- `frontend/pages/MatchDetail.tsx`

### Phase 4: Video Upload & Processing

**4.1 Upload Modal**

- Modal component
- Playsight link input
- Link validation
- Submit handler

**4.2 Player Identification**

- Frame extraction from Playsight video
- Multiple frame display
- Click-to-identify interface
- Coordinate capture
- Submit identification data

**4.3 Processing Status**

- Real-time progress updates via Supabase Realtime subscriptions
- Processing status display component
- Notification system (when complete) - Supabase Realtime or polling
- Queue management (start simple, move to Celery + Redis later if needed)

**4.4 Backend Integration**

- FastAPI endpoint for video submission
- Playsight link processing (research integration method)
- Frame extraction and storage
- Player identification storage in Supabase
- CV processing trigger (async task)
- Status updates via Supabase Realtime or polling

**Files to Create:**

- `frontend/components/upload/UploadModal.tsx` (React Hook Form + Zod)
- `frontend/components/upload/PlayerIdentification.tsx` (Multiple frame display)
- `frontend/components/upload/ProcessingStatus.tsx` (Realtime status updates)
- `backend/api/videos.py` (upload, status endpoints)
- `backend/services/playsight.py` (Playsight integration)
- `backend/services/cv_processor.py` (CV backend integration)
- `frontend/hooks/useVideoUpload.ts` (TanStack Query mutation)

### Phase 5: Team Management

**5.1 Team Code System**

- Team creation (coaches)
- Code generation
- Code sharing interface
- Player code entry/joining

**5.2 Team Views**

- Coach: See all team players
- Player: See team info
- Team member management

**Files to Create:**

- `frontend/components/team/TeamCode.tsx` (Code generation/entry)
- `frontend/components/team/TeamMembers.tsx` (Team member list)
- `backend/api/teams.py` (Team management endpoints)
- `frontend/hooks/useTeams.ts` (TanStack Query for team data)

### Phase 6: Backend API Development

**6.1 Authentication (Handled by Supabase)**

- Supabase Auth handles OAuth callbacks
- Session management via Supabase client
- User profile stored in Supabase `users` table
- Custom user metadata for role (coach/player)

**6.2 Match Endpoints**

- List matches (filtered by user role via RLS)
- Get match details
- Match data retrieval from Supabase
- Use Supabase client or REST API

**6.3 Stats Endpoints**

- Player stats aggregation (FastAPI endpoints)
- Season stats calculation
- Per-game stats
- Query Supabase for data, aggregate in FastAPI

**6.4 Video Processing Endpoints**

- Video upload/submission
- Processing status (update Supabase, use Realtime)
- Results retrieval

**Files to Create:**

- `frontend/lib/supabase/client.ts` (Supabase client)
- `frontend/lib/supabase/server.ts` (Server-side Supabase)
- `backend/api/matches.py` (Match endpoints)
- `backend/api/stats.py` (Stats aggregation endpoints)
- `backend/api/videos.py` (Video processing endpoints)
- `frontend/hooks/useAuth.ts` (Auth hooks with Supabase)

### Phase 7: CV Backend Integration

**7.1 Integration Layer**

- Connect FastAPI to existing tennis_analytics CV backend
- Video processing pipeline
- JSON output parsing
- Data storage

**7.2 Player Tracking**

- Color recognition implementation
- Player tracking throughout video
- Stats calculation per tracked player

**Files to Modify/Create:**

- `backend/services/cv_integration.py`
- Modify existing CV backend to output structured JSON
- `backend/services/player_tracker.py`

### Phase 8: Polish & Optimization

**8.1 UI/UX Enhancements**

- Loading states
- Error handling
- Responsive design
- Accessibility

**8.2 Performance**

- API optimization
- Frontend code splitting
- Caching strategies
- Database query optimization

**8.3 Testing**

- Unit tests for critical components
- Integration tests for API
- E2E tests for key flows

## Technical Stack (Selected)

### Frontend

- **Framework**: Next.js 14+ (App Router) - React with SSR, API routes, excellent DX
- **Styling**: Tailwind CSS + shadcn/ui components (copy-paste, highly customizable)
- **State Management**: 
  - TanStack Query (React Query) for server state
  - Zustand for client state (if needed)
- **Forms**: React Hook Form + Zod for validation
- **Charts**: Recharts (React-native, responsive, customizable)
- **Video**: react-player or native iframe for Playsight embedding
- **Database Client**: @supabase/supabase-js

### Backend

- **Framework**: FastAPI (Python) - matches existing CV backend
- **Database**: Supabase (PostgreSQL) - includes auth, realtime, storage
- **Auth**: Supabase Auth (OAuth, email/password, magic links)
- **Realtime**: Supabase Realtime subscriptions for processing status
- **Task Queue**: Start simple (in-memory or Supabase Edge Functions), move to Celery + Redis later if needed
- **File Storage**: Playsight (embedded), Supabase Storage for frames if needed

### Infrastructure

- **Development**: 
  - Local Next.js dev server
  - Local FastAPI server
  - Supabase local development (optional) or cloud project
- **Deployment**: 
  - Frontend: Vercel (optimized for Next.js)
  - Backend: Railway or Render (Python-friendly)
  - Database: Supabase (cloud)

## Key Integration Points

1. **Supabase Setup**: 
   - Create Supabase project
   - Configure OAuth providers (Google/Apple)
   - Set up database schema with RLS policies
   - Configure Supabase client in Next.js

2. **Playsight Integration**: Research Playsight API or scraping method for video extraction

3. **CV Backend**: Integrate existing `tennis_analytics` codebase for processing

4. **JSON Format**: Define schema for CV output (shot data, positions, timestamps) - store in Supabase `match_data` table

5. **Color Recognition**: Implement player tracking based on identification clicks

6. **Realtime Updates**: Use Supabase Realtime subscriptions for processing status updates

## Data Flow Example

```
User uploads Playsight link
  → Backend extracts video/frames
  → User identifies player in frames
  → Backend stores identification
  → CV processing triggered (async)
  → CV backend processes video
  → JSON output generated
  → JSON stored in database
  → Frontend fetches match data
  → Court diagram renders with shots
  → User clicks shot → Video panel opens at timestamp
```

## Next Steps After Plan Approval

1. Set up Supabase project and configure OAuth providers
2. Initialize Next.js 14+ frontend with Tailwind CSS + shadcn/ui
3. Set up FastAPI backend structure
4. Create Supabase database schema (tables, RLS policies)
5. Configure Supabase client in Next.js
6. Implement authentication flow with Supabase Auth
7. Build core UI components (layout, dashboard)
8. Integrate CV backend
9. Test end-to-end flow

## Development Approach

**Go slow, one step at a time:**
- Start with Phase 1: Project setup and Supabase configuration
- Get authentication working first
- Then build UI components incrementally
- Test each phase before moving to the next
- Focus on MVP features first, polish later