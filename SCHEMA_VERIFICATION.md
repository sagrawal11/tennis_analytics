# Database Schema Verification

## Verified: Schema Supports All Features

### ✅ Role Assignment
- **Table**: `public.users`
- **Field**: `role TEXT CHECK (role IN ('coach', 'player')) NOT NULL DEFAULT 'player'`
- **Status**: ✅ Supports role selection during signup
- **Note**: Database trigger currently sets default to 'player', but can be updated via user metadata during signup

### ✅ Profile Data
- **Table**: `public.users`
- **Fields**: `id`, `email`, `name`, `role`, `created_at`, `updated_at`
- **Teams**: Can be queried via `team_members` join
- **Status**: ✅ All basic profile data available
- **No changes needed**: All profile information can be queried from existing tables

### ✅ Team Management
- **Tables**: `public.teams`, `public.team_members`
- **Features**:
  - Teams have unique codes
  - Team members junction table links users to teams
  - RLS policies support coach/player access
- **Status**: ✅ Fully supported

### ✅ Match Ownership
- **Table**: `public.matches`
- **Field**: `user_id UUID REFERENCES public.users(id)`
- **Status**: ✅ Supports coach uploading for players
- **Note**: Backend API needs update to accept optional `user_id` parameter for coach uploads
- **Current**: Uses authenticated user's ID
- **Required**: Allow coaches to specify `user_id` of selected player

### ✅ Team Members Access
- **Table**: `public.team_members`
- **Fields**: `team_id`, `user_id`, `role`, `joined_at`
- **Status**: ✅ Can query team members for player dropdown
- **API Endpoint**: `/api/teams/{team_id}/members` exists

### ✅ Match Status
- **Table**: `public.matches`
- **Field**: `status TEXT CHECK (status IN ('pending', 'processing', 'completed', 'failed'))`
- **Status**: ✅ Supports conditional rendering based on status

## Backend API Updates

### ✅ 1. Match Creation Endpoint - COMPLETED
**File**: `backend/api/matches.py`

**Updated**:
- Added `user_id: Optional[str] = None` to `MatchCreate` model
- Updated `create_match` endpoint to:
  - Accept optional `user_id` in request
  - Verify user is coach if `user_id` provided
  - Verify target player is on coach's team
  - Use provided `user_id` for match ownership if valid

### ✅ 2. User Signup with Role - COMPLETED
**File**: `supabase/schema.sql` (trigger)

**Updated**:
- Modified `handle_new_user()` function to use role from metadata
- Uses `COALESCE(NEW.raw_user_meta_data->>'role', 'player')` instead of hardcoded 'player'

**File**: `frontend/hooks/useAuth.ts`

**Updated**:
- Added `role?: 'coach' | 'player'` parameter to `signUp` function
- Passes role to Supabase signup metadata: `role: role || 'player'`

## Summary

✅ **Schema**: No changes needed - all features supported  
✅ **Backend API**: All updates completed
1. ✅ Match creation endpoint accepts optional `user_id`
2. ✅ User signup trigger uses role from metadata
3. ✅ Frontend hook passes role in signup

All backend updates are complete and ready for use.
