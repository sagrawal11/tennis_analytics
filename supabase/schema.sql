-- Tennis Analytics Database Schema
-- Run this in Supabase SQL Editor after creating your project

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Teams table (created first since users references it)
CREATE TABLE public.teams (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    name TEXT NOT NULL,
    code TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW()) NOT NULL
);

-- Users table (extends Supabase auth.users)
-- Note: We don't create FK to auth.users here to avoid privilege issues
-- The trigger will populate this table with the auth.users.id
CREATE TABLE public.users (
    id UUID PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    name TEXT,
    role TEXT CHECK (role IN ('coach', 'player')) NOT NULL DEFAULT 'player',
    team_id UUID REFERENCES public.teams(id),
    activation_key TEXT,
    activated_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW()) NOT NULL
);

-- Team members junction table
CREATE TABLE public.team_members (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    team_id UUID REFERENCES public.teams(id) ON DELETE CASCADE NOT NULL,
    user_id UUID REFERENCES public.users(id) ON DELETE CASCADE NOT NULL,
    role TEXT CHECK (role IN ('coach', 'player')) NOT NULL,
    joined_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW()) NOT NULL,
    UNIQUE(team_id, user_id)
);

-- Matches table
CREATE TABLE public.matches (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES public.users(id) ON DELETE CASCADE NOT NULL,
    player_name TEXT,
    playsight_link TEXT NOT NULL,
    video_url TEXT,
    match_date DATE,
    opponent TEXT,
    notes TEXT,
    status TEXT CHECK (status IN ('pending', 'processing', 'completed', 'failed')) DEFAULT 'pending' NOT NULL,
    processed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW()) NOT NULL
);

-- Match data table (stores JSON output from CV backend)
CREATE TABLE public.match_data (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    match_id UUID REFERENCES public.matches(id) ON DELETE CASCADE NOT NULL,
    json_data JSONB NOT NULL,
    stats_summary JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW()) NOT NULL
);

-- Shots table
CREATE TABLE public.shots (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    match_id UUID REFERENCES public.matches(id) ON DELETE CASCADE NOT NULL,
    shot_type TEXT,
    start_pos JSONB NOT NULL, -- {x: number, y: number}
    end_pos JSONB NOT NULL, -- {x: number, y: number}
    timestamp INTEGER NOT NULL, -- Frame number or timestamp
    video_timestamp REAL, -- Video timestamp in seconds
    result TEXT CHECK (result IN ('winner', 'error', 'in_play')) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW()) NOT NULL
);

-- Player identifications table
CREATE TABLE public.player_identifications (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    match_id UUID REFERENCES public.matches(id) ON DELETE CASCADE NOT NULL,
    frame_data JSONB NOT NULL, -- Frame image data or reference
    selected_player_coords JSONB NOT NULL, -- {x: number, y: number} or bounding box
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW()) NOT NULL
);

-- Indexes for performance
CREATE INDEX idx_users_team_id ON public.users(team_id);
CREATE INDEX idx_users_activation_key ON public.users(activation_key);
CREATE INDEX idx_team_members_team_id ON public.team_members(team_id);
CREATE INDEX idx_team_members_user_id ON public.team_members(user_id);
CREATE INDEX idx_matches_user_id ON public.matches(user_id);
CREATE INDEX idx_matches_status ON public.matches(status);
CREATE INDEX idx_match_data_match_id ON public.match_data(match_id);
CREATE INDEX idx_shots_match_id ON public.shots(match_id);
CREATE INDEX idx_player_identifications_match_id ON public.player_identifications(match_id);

-- Row Level Security (RLS) Policies

-- Enable RLS on all tables
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.teams ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.team_members ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.matches ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.match_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.shots ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.player_identifications ENABLE ROW LEVEL SECURITY;

-- Users policies
CREATE POLICY "Users can view their own profile"
    ON public.users FOR SELECT
    USING (auth.uid() = id);

CREATE POLICY "Users can update their own profile"
    ON public.users FOR UPDATE
    USING (auth.uid() = id);

-- Teams policies
CREATE POLICY "Users can view teams they belong to"
    ON public.teams FOR SELECT
    USING (
        id IN (
            SELECT team_id FROM public.team_members
            WHERE user_id = auth.uid()
        )
    );

CREATE POLICY "Coaches can create teams"
    ON public.teams FOR INSERT
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.users
            WHERE id = auth.uid() AND role = 'coach'
        )
    );

-- Team members policies
CREATE POLICY "Users can view team members of their teams"
    ON public.team_members FOR SELECT
    USING (
        team_id IN (
            SELECT team_id FROM public.team_members
            WHERE user_id = auth.uid()
        )
    );

-- Helper function to check if user is coach on team
CREATE OR REPLACE FUNCTION public.is_coach_on_team(check_team_id UUID)
RETURNS BOOLEAN AS $$
BEGIN
  RETURN EXISTS (
    SELECT 1 
    FROM public.team_members tm
    JOIN public.users u ON u.id = tm.user_id
    WHERE tm.team_id = check_team_id
    AND tm.user_id = auth.uid()
    AND u.role = 'coach'
  );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE POLICY "Coaches can add team members"
    ON public.team_members FOR INSERT
    WITH CHECK (public.is_coach_on_team(team_id));

-- Policy for players to join teams using team code
-- Players can add themselves to a team (application will look up team by code first)
CREATE POLICY "Players can join teams with code"
    ON public.team_members FOR INSERT
    WITH CHECK (
        -- User must be adding themselves
        auth.uid() = user_id
        -- User must be joining as a player
        AND role = 'player'
        -- Team must exist (validated by foreign key)
        AND EXISTS (
            SELECT 1 FROM public.teams
            WHERE id = team_id
        )
    );

-- Matches policies
CREATE POLICY "Users can view their own matches"
    ON public.matches FOR SELECT
    USING (user_id = auth.uid());

CREATE POLICY "Coaches can view all team member matches"
    ON public.matches FOR SELECT
    USING (
        -- Check if the viewing user is a coach
        EXISTS (
            SELECT 1 FROM public.users
            WHERE id = auth.uid() AND role = 'coach'
        )
        -- AND the match owner is on the same team as the coach
        AND EXISTS (
            SELECT 1 FROM public.team_members tm1
            JOIN public.team_members tm2 ON tm1.team_id = tm2.team_id
            WHERE tm1.user_id = auth.uid()  -- Coach viewing
            AND tm2.user_id = matches.user_id  -- Match owner
        )
    );

CREATE POLICY "Users can create their own matches"
    ON public.matches FOR INSERT
    WITH CHECK (user_id = auth.uid());

CREATE POLICY "Users can update their own matches"
    ON public.matches FOR UPDATE
    USING (user_id = auth.uid());

-- Match data policies (same as matches)
CREATE POLICY "Users can view match data for their matches"
    ON public.match_data FOR SELECT
    USING (
        match_id IN (
            SELECT id FROM public.matches
            WHERE user_id = auth.uid()
            OR (
                -- Coach can view if match owner is on their team
                EXISTS (
                    SELECT 1 FROM public.users
                    WHERE id = auth.uid() AND role = 'coach'
                )
                AND EXISTS (
                    SELECT 1 FROM public.team_members tm1
                    JOIN public.team_members tm2 ON tm1.team_id = tm2.team_id
                    JOIN public.matches m ON m.user_id = tm2.user_id
                    WHERE tm1.user_id = auth.uid()
                    AND m.id = match_data.match_id
                )
            )
        )
    );

-- Shots policies (same as matches)
CREATE POLICY "Users can view shots for their matches"
    ON public.shots FOR SELECT
    USING (
        match_id IN (
            SELECT id FROM public.matches
            WHERE user_id = auth.uid()
            OR (
                -- Coach can view if match owner is on their team
                EXISTS (
                    SELECT 1 FROM public.users
                    WHERE id = auth.uid() AND role = 'coach'
                )
                AND EXISTS (
                    SELECT 1 FROM public.team_members tm1
                    JOIN public.team_members tm2 ON tm1.team_id = tm2.team_id
                    JOIN public.matches m ON m.user_id = tm2.user_id
                    WHERE tm1.user_id = auth.uid()
                    AND m.id = shots.match_id
                )
            )
        )
    );

-- Player identifications policies
CREATE POLICY "Users can view their own player identifications"
    ON public.player_identifications FOR SELECT
    USING (
        match_id IN (
            SELECT id FROM public.matches WHERE user_id = auth.uid()
        )
    );

CREATE POLICY "Users can create player identifications for their matches"
    ON public.player_identifications FOR INSERT
    WITH CHECK (
        match_id IN (
            SELECT id FROM public.matches WHERE user_id = auth.uid()
        )
    );

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = TIMEZONE('utc', NOW());
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers to automatically update updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON public.users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_teams_updated_at BEFORE UPDATE ON public.teams
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_matches_updated_at BEFORE UPDATE ON public.matches
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_match_data_updated_at BEFORE UPDATE ON public.match_data
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to automatically create user profile when auth user is created
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
DECLARE
  user_role TEXT;
BEGIN
  -- Extract role from metadata, ensuring it's a valid value
  user_role := COALESCE(
    NULLIF(TRIM(NEW.raw_user_meta_data->>'role'), ''),
    'player'
  );
  
  -- Ensure role is valid (coach or player)
  IF user_role NOT IN ('coach', 'player') THEN
    user_role := 'player';
  END IF;
  
  INSERT INTO public.users (id, email, name, role)
  VALUES (
    NEW.id,
    NEW.email,
    COALESCE(NEW.raw_user_meta_data->>'name', ''),
    user_role
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger to create user profile on signup
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- Comments for activation key columns
COMMENT ON COLUMN public.users.activation_key IS 'Activation key for coaches to unlock platform access';
COMMENT ON COLUMN public.users.activated_at IS 'Timestamp when activation key was validated';
