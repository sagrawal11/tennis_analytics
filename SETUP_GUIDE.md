# Tennis Analytics Setup Guide

## Step 1: Supabase Setup

### 1.1 Create Supabase Project

1. Go to [https://supabase.com](https://supabase.com) and sign up/login
2. Click "New Project"
3. Fill in:
   - **Project Name**: `tennis-analytics` (or your preferred name)
   - **Database Password**: Create a strong password (save this!)
   - **Region**: Choose closest to you
   - **Pricing Plan**: Free tier is fine for development
4. Wait for project to initialize (takes ~2 minutes)

### 1.2 Get API Credentials

1. Go to **Settings** → **API**
2. Copy the following:
   - **Project URL** (e.g., `https://xxxxx.supabase.co`)
   - **anon/public key** (starts with `eyJ...`)
   - **service_role key** (keep this secret! Only for backend)

### 1.3 Configure OAuth Providers

1. Go to **Authentication** → **Providers**
2. Enable **Google**:
   - Click "Google"
   - Toggle "Enable Google provider"
   - You'll need to create OAuth credentials in Google Cloud Console
   - Add authorized redirect URI: `https://your-project.supabase.co/auth/v1/callback`
3. Enable **Apple** (optional):
   - Similar process, requires Apple Developer account

### 1.4 Set Up Database Schema

Run the SQL script in `supabase/schema.sql` (will be created in next steps) in the Supabase SQL Editor.

## Step 2: Environment Variables

After getting your Supabase credentials, create `.env.local` files:

- `frontend/.env.local` - Frontend environment variables
- `backend/.env` - Backend environment variables

See `.env.example` files for required variables.
