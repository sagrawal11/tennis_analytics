# v0 Detailed Integration Prompt - Complete Specification

## COPY THIS ENTIRE DOCUMENT INTO v0

---

# Courtvision - Tennis Analytics Platform - Complete UI Specification

Create a modern, beautiful tennis analytics platform called "Courtvision" using Next.js 16 App Router, React 19, TypeScript, Tailwind CSS 4.0, and shadcn/ui components.

**Brand Name**: Courtvision  
**Logo**: Minimalist drawing of an owl's eyes (will be provided as image file - place in `public/logo.png` or `public/logo.svg`, use Next.js Image component)  
**Color Scheme**: 
- **Background**: Black (#000000 or #0a0a0a)
- **Primary Accent**: Emerald Green (#50C878)
- **Highlights**: White (#FFFFFF)
- **Cards**: Dark gray (#1a1a1a or #262626)
- **Borders**: Gray (#333333 or #808080)

---

## KEY DESIGN CHANGES

### Landing Page (NEW)
- Beautiful marketing page at `/` (root route)
- Hero section with video/image background
- Features section with 6 feature cards
- Testimonials section with 3+ testimonials
- Footer with links
- Sign In/Sign Up buttons in top right navigation
- Auth happens via modal (no separate `/login` page)

### Logo Placement
- **Landing Page**: Logo + "Courtvision" in top left of navigation bar
- **Dashboard/Sidebar**: Logo + "Courtvision" at top of left sidebar

### Color Theme
- **Dark theme throughout**: Black backgrounds, dark gray cards
- **Emerald green accents**: #50C878 for buttons, highlights, active states
- **White text**: On dark backgrounds
- **Subtle borders**: Gray borders for separation

### Authentication Flow
- Landing page is public (anyone can view)
- Sign In/Sign Up buttons open modal
- Modal contains login/signup form (same as previous login page)
- After successful auth, redirect to `/dashboard`
- All protected pages use dark theme with emerald green accents

---

## TECH STACK REQUIREMENTS

- **Framework**: Next.js 16.0.10 (App Router)
- **React**: 19.0.0
- **TypeScript**: 5.7.2
- **Styling**: Tailwind CSS 4.0.0
- **UI Components**: shadcn/ui (Radix UI primitives)
- **Icons**: lucide-react (use icons like Home, BarChart3, Users, LogOut, Plus, X, etc.)
- **State Management**: TanStack Query 5.90.12 for server state
- **Authentication**: Supabase Auth
- **Fonts**: Geist Sans and Geist Mono (already configured)

---

## EXISTING HOOKS (MUST USE - DO NOT CREATE NEW ONES)

### `useAuth()` Hook
```typescript
// Location: hooks/useAuth.ts
// Import: import { useAuth } from '@/hooks/useAuth'

const { 
  signUp,      // (email: string, password: string, name?: string, role?: 'coach' | 'player') => Promise<{data, error}>
  signIn,      // (email: string, password: string) => Promise<{data, error}>
  signOut,     // () => Promise<void>
  getUser,     // () => Promise<User | null>
  getSession,  // () => Promise<Session | null>
  loading      // boolean
} = useAuth()
```

### `useMatches()` Hook
```typescript
// Location: hooks/useMatches.ts
// Import: import { useMatches } from '@/hooks/useMatches'

const { 
  data: matches,    // Match[] | undefined
  isLoading,        // boolean
  error            // Error | null
} = useMatches()
```

### `useTeams()` Hook
```typescript
// Location: hooks/useTeams.ts
// Import: import { useTeams } from '@/hooks/useTeams'

const {
  teams,           // Team[]
  isLoading,       // boolean
  createTeam,      // (name: string) => Promise<any>
  joinTeam,        // (code: string) => Promise<any>
  isCreating,      // boolean
  isJoining        // boolean
} = useTeams()
```

---

## TYPE DEFINITIONS

```typescript
interface User {
  id: string                    // UUID
  email: string
  name: string | null
  role: 'coach' | 'player'
  team_id: string | null       // UUID
  created_at: string            // ISO timestamp
  updated_at: string            // ISO timestamp
}

interface Team {
  id: string                    // UUID
  name: string
  code: string                  // Unique 6-character alphanumeric code
  created_at: string            // ISO timestamp
  updated_at: string            // ISO timestamp
}

interface Match {
  id: string                    // UUID
  user_id: string               // UUID
  player_name: string | null
  playsight_link: string
  video_url: string | null
  status: 'pending' | 'processing' | 'completed' | 'failed'
  processed_at: string | null   // ISO timestamp
  created_at: string            // ISO timestamp
  updated_at: string            // ISO timestamp
}

interface Shot {
  id: string                    // UUID
  match_id: string              // UUID
  shot_type: string | null
  start_pos: { x: number; y: number }  // Normalized 0-1 coordinates
  end_pos: { x: number; y: number }    // Normalized 0-1 coordinates
  timestamp: number              // Frame number
  video_timestamp: number | null // Seconds
  result: 'winner' | 'error' | 'in_play'
  created_at: string            // ISO timestamp
}
```

---

## LAYOUT STRUCTURE

### MainLayout Component
**Every protected page must be wrapped in MainLayout:**

```typescript
import { MainLayout } from '@/components/layout/MainLayout'

export default function Page() {
  return (
    <MainLayout>
      {/* Page content */}
    </MainLayout>
  )
}
```

**MainLayout includes:**
1. **Left Sidebar** (fixed, 256px wide, dark background)
2. **Main Content Area** (flex-1, scrollable)
3. **Floating Action Button** (fixed bottom-right, circular "+" button)

---

## PAGE-BY-PAGE SPECIFICATIONS

---

### 1. LANDING PAGE (`/`) - NEW

**Route**: `app/page.tsx`  
**Type**: Public page (no MainLayout)  
**Purpose**: Beautiful marketing/landing page with hero section, features, testimonials

#### Layout
- Full-width, dark theme (black background)
- Smooth scrolling sections
- Modern, clean design

#### Top Navigation Bar
- **Position**: Fixed at top, z-50, backdrop blur
- **Background**: Black with slight transparency (bg-black/80 or bg-black/90)
- **Height**: h-16 or h-20
- **Padding**: px-6 or px-8
- **Layout**: Flex, justify-between, items-center

**Left Side:**
- **Logo + Brand Name**:
  - Flex items-center gap-3
  - Logo image: Minimalist owl's eyes drawing (use Next.js Image component: `import Image from 'next/image'`)
  - Logo path: `/logo.png` or `/logo.svg` (in public folder)
  - Logo size: w-10 h-10 or w-12 h-12
  - Brand name: "Courtvision" (text-xl or text-2xl, font-bold, text-white)
  - Hover effect: Slight scale or opacity change
  - Optional: Link to home page (`/`)

**Right Side:**
- **Auth Buttons**:
  - Flex items-center gap-4
  - "Sign In" button:
    - Variant: ghost or outline
    - Text: white or emerald green
    - On click: Opens login modal
  - "Sign Up" button:
    - Variant: default
    - Background: emerald green (#50C878)
    - Text: black or white (high contrast)
    - On click: Opens signup modal (or same modal, toggle to signup)

#### Hero Section
- **Full viewport height**: min-h-screen
- **Background**: 
  - Option 1: Hero video (tennis match footage, muted, autoplay, loop)
  - Option 2: Hero image (tennis court, player action shot)
  - Overlay: Black gradient overlay (bg-black/60 or bg-black/70) for text readability
- **Content**: Centered, max-width container
- **Layout**: Flex flex-col, justify-center, items-center, text-center
- **Padding**: px-4, py-20 or py-32

**Hero Content:**
- **Headline**: 
  - Text: "Elevate Your Tennis Game with AI-Powered Analytics" (or similar)
  - Size: text-5xl or text-6xl, font-bold
  - Color: white
  - Margin: mb-6
- **Subheadline**:
  - Text: "Track every shot, analyze every match, improve every day"
  - Size: text-xl or text-2xl
  - Color: light gray (#e5e5e5)
  - Margin: mb-8
- **CTA Buttons**:
  - Flex gap-4, justify-center
  - Primary CTA: "Get Started" or "Start Free Trial"
    - Background: emerald green (#50C878)
    - Text: black or white
    - Size: lg
    - On click: Opens signup modal
  - Secondary CTA: "Watch Demo" or "Learn More"
    - Variant: outline
    - Border: white or emerald green
    - Text: white
    - On click: Scrolls to features section or opens demo video

#### Features Section
- **Background**: Dark gray (#1a1a1a) or black
- **Padding**: py-20 or py-24
- **Container**: max-w-7xl, mx-auto, px-4

**Section Header:**
- Title: "Everything You Need to Improve" (text-4xl, font-bold, text-white, text-center, mb-4)
- Subtitle: "Powerful analytics tools for coaches and players" (text-xl, text-gray-400, text-center, mb-16)

**Features Grid:**
- Grid: grid-cols-1 md:grid-cols-2 lg:grid-cols-3, gap-8
- Each feature card:
  - Background: Dark gray (#262626) or slightly lighter
  - Padding: p-6 or p-8
  - Rounded: rounded-lg or rounded-xl
  - Border: Optional subtle border (border border-gray-800)
  - Hover: Slight scale or border color change (emerald green)
  - Content:
    - Icon: Large icon (lucide-react) in emerald green, mb-4
    - Title: text-xl, font-semibold, text-white, mb-2
    - Description: text-gray-400, text-sm or text-base

**Feature Examples:**
1. **Shot Analysis**
   - Icon: Target or Crosshair
   - Title: "Precise Shot Tracking"
   - Description: "Every shot mapped with AI-powered accuracy"

2. **Performance Insights**
   - Icon: BarChart3 or TrendingUp
   - Title: "Performance Analytics"
   - Description: "Deep insights into your game patterns"

3. **Team Management**
   - Icon: Users or UserCheck
   - Title: "Team Collaboration"
   - Description: "Coaches and players working together"

4. **Real-time Processing**
   - Icon: Zap or Activity
   - Title: "Instant Analysis"
   - Description: "Get insights in minutes, not hours"

5. **Visual Court Diagram**
   - Icon: Layout or Grid
   - Title: "Interactive Court View"
   - Description: "See your shots visualized on the court"

6. **Video Integration**
   - Icon: Video or Play
   - Title: "Video Playback"
   - Description: "Watch key moments from your matches"

#### Testimonials Section
- **Background**: Black or very dark gray
- **Padding**: py-20 or py-24
- **Container**: max-w-7xl, mx-auto, px-4

**Section Header:**
- Title: "Trusted by Coaches and Players" (text-4xl, font-bold, text-white, text-center, mb-4)
- Subtitle: "See what our users are saying" (text-xl, text-gray-400, text-center, mb-16)

**Testimonials Grid:**
- Grid: grid-cols-1 md:grid-cols-2 lg:grid-cols-3, gap-8
- Each testimonial card:
  - Background: Dark gray (#262626)
  - Padding: p-6 or p-8
  - Rounded: rounded-lg or rounded-xl
  - Border: Optional subtle border
  - Content:
    - Quote: text-lg, text-white, italic, mb-4
    - Author section: Flex items-center gap-4
      - Avatar: Circular, bg-emerald-green/20, w-12 h-12, flex items-center justify-center
      - Author info:
        - Name: text-white, font-semibold
        - Role: text-gray-400, text-sm (e.g., "Head Coach, Stanford University")

**Testimonial Examples (Placeholder - can be updated later):**
1. "Courtvision has transformed how we analyze our players' performance. The insights are incredible."
   - Author: "Sarah Johnson", "Head Coach"

2. "As a player, seeing my shot patterns visualized helps me understand my game so much better."
   - Author: "Michael Chen", "Professional Player"

3. "The team management features make it easy to track all my players in one place."
   - Author: "David Martinez", "College Coach"

#### Footer Section
- **Background**: Very dark gray or black
- **Padding**: py-12
- **Container**: max-w-7xl, mx-auto, px-4
- **Layout**: Grid, grid-cols-1 md:grid-cols-4, gap-8

**Footer Content:**
- **Brand Column**:
  - Logo + "Courtvision"
  - Tagline: "AI-Powered Tennis Analytics"
  - Social links (optional)
- **Product Column**:
  - Title: "Product" (text-white, font-semibold, mb-4)
  - Links: Features, Pricing, Demo (text-gray-400, hover:text-emerald-green)
- **Company Column**:
  - Title: "Company" (text-white, font-semibold, mb-4)
  - Links: About, Blog, Careers (text-gray-400, hover:text-emerald-green)
- **Support Column**:
  - Title: "Support" (text-white, font-semibold, mb-4)
  - Links: Help Center, Contact, Privacy (text-gray-400, hover:text-emerald-green)

**Bottom Bar:**
- Border-top: border-gray-800
- Padding: py-6
- Text: "© 2024 Courtvision. All rights reserved." (text-gray-500, text-center)

#### Authentication Modal (NEW)

**Component**: `components/auth/AuthModal.tsx`  
**Props**: `{ isOpen: boolean, onClose: () => void, initialMode?: 'signin' | 'signup' }`

**Trigger**: 
- "Sign In" or "Sign Up" buttons in top nav or hero section
- Opens modal overlay

**Modal Overlay:**
- Fixed inset-0
- Background: bg-black/80 or bg-black/90 (backdrop blur)
- z-50
- Flex items-center justify-center
- Padding: p-4

**Modal Content:**
- Background: Dark gray (#1a1a1a or #262626)
- Max width: max-w-md
- Width: w-full
- Rounded: rounded-xl or rounded-2xl
- Padding: p-8
- Border: Optional subtle border (border border-gray-800)
- Shadow: Large shadow (shadow-2xl)

**Modal Header:**
- Flex justify-between items-center, mb-6
- Title: "Sign in" or "Create account" (text-2xl, font-bold, text-white)
- Close button: X icon (lucide-react), ghost variant, text-gray-400, hover:text-white
  - On click: Calls `onClose()`

**Modal Form:**
- Same form fields as previous login page specification:
  - Name input (signup only)
  - Role selection (signup only) - Radio buttons: "Player" (default) or "Coach"
  - Email input
  - Password input
- Styling adapted for dark theme:
  - Inputs: bg-black/50, border-gray-700, text-white, placeholder-gray-500
  - Focus: border-emerald-green, ring-emerald-green
  - Labels: text-gray-300

**Modal Buttons:**
- Submit button:
  - Background: emerald green (#50C878)
  - Text: black or white (high contrast)
  - Full width
  - On click: Same logic as previous login page
- Toggle button:
  - Text: "Don't have an account? Sign up" or "Already have an account? Sign in"
  - Color: emerald green (#50C878)
  - Hover: Lighter shade
  - On click: Toggles between signin/signup mode

**Modal Footer:**
- Error display: bg-red-900/20, border-red-800, text-red-300 (if error exists)

**State Management:**
- Same as previous login page
- `isSignUp`: boolean (toggles between sign in/sign up)
- `email`, `password`, `name`, `role`: controlled inputs
- `error`: string | null
- `loading`: from useAuth hook

**Success Behavior:**
- On successful signup/signin: Close modal, redirect to `/dashboard`

---

### 2. LOGIN PAGE (`/login`) - DEPRECATED

**Note**: This route is now deprecated. Authentication happens via modal on landing page. However, if users navigate directly to `/login`, redirect them to `/` (landing page) or show the same modal.

**Route**: `app/login/page.tsx`  
**Type**: Public page (redirects to `/` or shows auth modal)

#### Form Fields

**Sign Up Mode (when toggle is on):**
1. **Name Input**
   - Label: "Full name" (visually hidden, use sr-only)
   - Type: text
   - Placeholder: "Full name"
   - Required: true
   - Styling: border-gray-300, rounded-md, px-3 py-2

2. **Role Selection (NEW - REQUIRED)**
   - Label: "I am a..." (block, text-sm, font-medium, text-gray-700, mb-2)
   - Radio button group or toggle switch
   - Options:
     - "Player" (default, selected)
     - "Coach"
   - Required: true
   - State: `role` (string: 'player' | 'coach')
   - Default: 'player'
   - Styling: Use shadcn/ui RadioGroup or similar component

3. **Email Input**
   - Label: "Email address" (visually hidden)
   - Type: email
   - Placeholder: "Email address"
   - Required: true
   - AutoComplete: "email"

4. **Password Input**
   - Label: "Password" (visually hidden)
   - Type: password
   - Placeholder: "Password"
   - Required: true
   - AutoComplete: "new-password" (sign up) or "current-password" (sign in)

**Sign In Mode:**
- Only Email and Password fields (no Name field)

#### Buttons

1. **Submit Button**
   - Text: "Sign in" (sign in mode) or "Sign up" (sign up mode)
   - Loading text: "Loading..."
   - Disabled when: `loading` from useAuth is true
   - Style: Full width, indigo-600 background, white text, rounded-md
   - On click: Calls `handleSubmit` which:
     - Prevents default form submission
     - Clears any previous errors
     - If sign up: Calls `signUp(email, password, name, role)`
       - **NEW**: Passes `role` ('coach' or 'player') to signUp function
       - If error: Shows error message
       - If success AND `data.session` exists: Redirects to `/dashboard` (email confirmations disabled)
       - If success AND no session: Shows alert "Please check your email to confirm your account!" (email confirmations enabled)
       - **Note**: Backend/database trigger will set user role based on `role` parameter
     - If sign in: Calls `signIn(email, password)`
       - If error: Shows error message
       - If success: Redirects to `/dashboard`

2. **Toggle Button** (below submit button)
   - Text: "Don't have an account? Sign up" (sign in mode) or "Already have an account? Sign in" (sign up mode)
   - Style: Text button, indigo-600 color, hover: indigo-500
   - On click: Toggles `isSignUp` state, clears error

#### Error Display
- If error exists: Red background box (bg-red-50, border-red-200)
- Shows error message from `error.message`
- Positioned above submit button

#### State Management
- `email`: string (controlled input)
- `password`: string (controlled input)
- `name`: string (controlled input, only in sign up mode)
- `role`: 'player' | 'coach' (controlled, defaults to 'player', only in sign up mode)
- `isSignUp`: boolean (toggles between sign in/sign up)
- `error`: string | null (error message)
- `loading`: from useAuth hook

---

### 2. DASHBOARD PAGE (`/dashboard`)

**Route**: `app/dashboard/page.tsx`  
**Type**: Protected page (wrapped in MainLayout)  
**Purpose**: Display user's matches grouped by date

#### Layout
- Background: Black or very dark gray (matches overall theme)
- Container: mx-auto, px-4, py-8
- Title: "Dashboard" (text-3xl, font-bold, mb-6, text-white)

#### Loading State
- If `isLoading` from useMatches is true:
  - Show: "Loading matches..." (centered, text-gray-400, py-12)

#### Error State
- If `error` exists:
  - Show: Red alert box (bg-red-900/20, border-red-800, border-2, rounded-lg, p-4)
  - Message: "Error loading matches. Please try again." (text-red-400)

#### Empty State (UPDATED - Onboarding Messages)
- If `matches` is empty or length === 0:
  - Dark card (bg-gray-900 or #1a1a1a, rounded-lg, shadow-xl, border border-gray-800, p-12, text-center)
  - **Onboarding Message** (not just empty state):
    - Title: "Welcome to Courtvision!" (text-xl, font-semibold, mb-2, text-white)
    - Message: "Upload your first match to see detailed analytics and insights." (text-gray-400, mb-4)
    - Helper: "Click the + button in the bottom right to get started." (text-sm, text-gray-500)
    - Optional: Add illustration or icon in emerald green

#### Matches Display

**Grouping Logic:**
- Group matches by date using `created_at` field
- Format date: `new Date(match.created_at).toLocaleDateString()`
- Sort dates: Most recent first

**Date Group UI:**
- Each date group is a dark card (bg-gray-900 or #1a1a1a, rounded-lg, shadow-xl, border border-gray-800)
- Date header button:
  - Full width, flex layout (justify-between, items-center)
  - Padding: px-6 py-4
  - Hover: bg-gray-800/50
  - Left side: Date text (text-lg, font-semibold, text-white)
  - Right side: Match count (text-sm, text-gray-400) - e.g., "3 matches" or "1 match"
  - On click: Toggles expansion (if already expanded, collapses; if collapsed, expands)
  - State: `expandedDate` (string | null) - tracks which date is expanded

**Expanded Content:**
- Only shows when `expandedDate === date`
- Padding: px-6 pb-4
- Grid layout: grid-cols-1 md:grid-cols-2 lg:grid-cols-3, gap-4
- Renders MatchCard components for each match in that date group

#### MatchCard Component
- Link wrapper: `href={`/matches/${match.id}`}`
- Card: bg-gray-900 or #1a1a1a, rounded-lg, shadow-xl, border border-gray-800, p-6
- Hover: border-emerald-green/50, shadow-emerald-green/10, transition-all, cursor-pointer
- Content:
  - Top row: Flex justify-between
    - Left: Match name (text-lg, font-semibold, text-white) - shows `match.player_name || 'Match'`
    - Right: Status badge (px-2, py-1, rounded, text-xs, font-medium)
      - Colors (on dark background):
        - pending: bg-amber-900/30, text-amber-400, border border-amber-800
        - processing: bg-blue-900/30, text-blue-400, border border-blue-800
        - completed: bg-emerald-900/30, text-emerald-400, border border-emerald-800
        - failed: bg-red-900/30, text-red-400, border border-red-800
  - Date: text-sm, text-gray-400, mb-2 - formatted `created_at`
  - Playsight link: text-xs, text-gray-500, truncate - shows `playsight_link`

#### Coach vs Player View (UPDATED)

**Players:**
- See only their own matches
- No filter needed
- Grouped by date only

**Coaches (UPDATED - Player Filter):**
- See all matches from all team members
- **NEW**: Player filter dropdown at top of dashboard (below title)
  - Position: Below "Dashboard" title, mb-4
  - Label: "Filter by player:" (text-sm, font-medium, text-gray-700, mr-2)
  - Dropdown: Select component (shadcn/ui Select)
  - Options:
    - "All Players" (default, selected)
    - List of team member names (fetched from teams API)
  - State: `selectedPlayerId` (string | null, null = "All Players")
  - On change: Filters matches to show only selected player's matches
  - When "All Players" selected: Shows all team matches (current behavior)
  - When specific player selected: Shows only that player's matches, still grouped by date
- Logic: Frontend filters the matches array based on `selectedPlayerId`
- **Data Source**: Need to fetch team members list (can use teams API or create new hook)

---

### 3. TEAMS PAGE (`/teams`)

**Route**: `app/teams/page.tsx`  
**Type**: Protected page (wrapped in MainLayout)  
**Purpose**: Team management (create/join teams)

#### Layout
- Background: Black or very dark gray (matches overall theme)
- Container: mx-auto, px-4, py-8
- Title: "Teams" (text-3xl, font-bold, mb-6, text-white)

#### Coach View (when `profile.role === 'coach'`)

**Create Team Section:**
- Dark card (bg-gray-900 or #1a1a1a, rounded-lg, shadow-xl, border border-gray-800, p-6)
- Title: "Create Team" (text-xl, font-semibold, mb-4, text-white)

**Empty State (Onboarding):**
- If `teams.length === 0` and `!showCreate`:
  - Onboarding message: "Welcome! Get started by creating your first team." (text-gray-600, mb-4)
  - Helper: "Teams help you organize and track your players' matches." (text-sm, text-gray-500, mb-4)

**State: `showCreate` (boolean)**
- If `showCreate === false`:
  - Shows button: "Create New Team"
  - Button style: Default shadcn Button
  - On click: Sets `showCreate` to true
- If `showCreate === true`:
  - Shows CreateTeam component
  - CreateTeam handles its own state and UI

**Your Teams Section:**
- Only shows if `!isLoading && teams.length > 0`
- Title: "Your Teams" (implied, or add if needed)
- Grid/List: space-y-4
- Each team card:
  - Dark card (bg-gray-900 or #1a1a1a, rounded-lg, shadow-xl, border border-gray-800, p-6)
  - Top row: Flex justify-between, items-center, mb-4
    - Left: Team name (text-lg, font-semibold, text-white) - `team.name`
    - Right: Team code badge (bg-black/50, border border-emerald-500/50, rounded, px-3, py-1)
      - Code text: text-sm, font-mono, font-semibold, text-emerald-400 - `team.code`
  - Below: "Team Members" section
    - Title: "Team Members" (text-sm, font-medium, text-gray-400, mb-2)
    - TeamMembers component: `<TeamMembers teamId={team.id} />`

#### Player View (when `profile.role === 'player'`)

**Join Team Section:**
- Dark card (bg-gray-900 or #1a1a1a, rounded-lg, shadow-xl, border border-gray-800, p-6)
- Title: "Join Team" (text-xl, font-semibold, mb-4, text-white)

**Empty State (Onboarding):**
- If `teams.length === 0`:
  - Onboarding message: "Join a team to get started!" (text-gray-600, mb-4)
  - Helper: "Ask your coach for a team code to join." (text-sm, text-gray-500, mb-4)

- TeamCode component: `<TeamCode onJoin={() => {}} />`

**Your Teams Section:**
- Only shows if `!isLoading && teams.length > 0`
- Title: "Your Teams" (text-lg, font-semibold, mb-4)
- List: space-y-2
- Each team: Simple card (p-3, bg-black/50, border border-gray-800, rounded-lg)
  - Team name: font-medium, text-white - `team.name`

---

### 4. CREATE TEAM COMPONENT

**Location**: `components/team/CreateTeam.tsx`  
**Props**: `{ onCreated?: () => void }`

#### State
- `teamName`: string
- `teamCode`: string | null (null initially, set after successful creation)
- `error`: string | null
- `isCreating`: from useTeams hook

#### Success State (when `teamCode` is not null)
- Success box (bg-emerald-900/20, border-emerald-800, border-2, rounded-lg, p-4)
- Message: "Team created successfully!" (text-sm, font-medium, text-emerald-400, mb-2)
- Helper: "Share this code with your players:" (text-sm, text-gray-400, mb-4)
- Code display: Dark box (bg-black/50, rounded, border-2 border-emerald-500, p-4, text-center)
  - Code: text-3xl, font-bold, text-emerald-400, font-mono - `teamCode`
  - **Note**: Code is displayed prominently, no copy button (players will type it in)
- Button: "Create Another Team" (variant="outline", border-gray-700, text-white, hover:border-emerald-500)
  - On click: Sets `teamCode` to null, calls `onCreated?.()`

#### Form State (when `teamCode === null`)
- Input field:
  - Label: "Team Name" (block, text-sm, font-medium, text-gray-400, mb-2)
  - Input: text type, full width, rounded-md, bg-black/50, border-gray-700, text-white, placeholder-gray-500, px-3 py-2, text-sm
  - Placeholder: "e.g., Varsity Tennis Team"
  - Focus: border-emerald-green, ring-emerald-green
  - Value: `teamName` (controlled)
  - On change: Updates `teamName`
- Error display:
  - If `error` exists: text-sm, text-red-400 - shows error message
- Submit button: "Create Team" (or "Creating..." if `isCreating`)
  - Background: emerald green (#50C878)
  - Text: black or white (high contrast)
  - Disabled: when `isCreating` is true
  - On click: `handleCreate` function:
    - Validates: If `teamName.trim()` is empty, sets error "Please enter a team name" and returns
    - Clears error
    - Calls `createTeam(teamName)` from useTeams
    - On success: Sets `teamCode` to `data.code`, clears `teamName`, calls `onCreated?.()`
    - On error: Sets error to `err.message || 'Failed to create team'`

---

### 5. TEAM CODE COMPONENT (Join Team)

**Location**: `components/team/TeamCode.tsx`  
**Props**: `{ onJoin?: () => void }`

#### State
- `code`: string (automatically uppercased)
- `error`: string | null
- `isJoining`: from useTeams hook

#### UI
- Label: "Enter Team Code" (block, text-sm, font-medium, text-gray-400, mb-2)
- Input container: Flex gap-2
  - Input field:
    - Type: text
    - Value: `code` (controlled)
    - Placeholder: "ABC123"
    - MaxLength: 6
    - Class: flex-1, rounded-md, bg-black/50, border-gray-700, text-white, placeholder-gray-500, px-3 py-2, text-sm, uppercase
    - Focus: border-emerald-green, ring-emerald-green
    - On change: Sets `code` to `e.target.value.toUpperCase()` (auto-uppercase)
  - Button: "Join" (or "Joining..." if `isJoining`)
    - Background: emerald green (#50C878)
    - Text: black or white (high contrast)
    - Disabled: when `isJoining` is true
    - On click: `handleJoin` function:
      - Validates: If `code.trim()` is empty, sets error "Please enter a team code" and returns
      - Clears error
      - Calls `joinTeam(code.toUpperCase())` from useTeams
      - On success: Clears `code`, calls `onJoin?.()`
      - On error: Sets error to `err.message || 'Failed to join team'`
- Error display:
  - If `error` exists: mt-2, text-sm, text-red-400 - shows error message

---

### 6. MATCH DETAIL PAGE (`/matches/[id]`)

**Route**: `app/matches/[id]/page.tsx`  
**Type**: Protected page (wrapped in MainLayout)  
**Purpose**: Display match visualization with court diagram, video, and stats

#### Data Fetching
- Server component fetches:
  - Match data from Supabase
  - Match data (JSON) from Supabase
  - Shots array from Supabase
- Passes to client component: `MatchDetailContent`

#### Layout
- Background: Black or very dark gray
- Container: mx-auto, px-4, py-8

#### Header Section
- Title: `{match.player_name || 'Match'} - {new Date(match.created_at).toLocaleDateString()}`
  - Style: text-3xl, font-bold, mb-2, text-white
- Processing Status:
  - Only shows if `match.status !== 'completed'`
  - Component: `<ProcessingStatus matchId={match.id} />`
  - Positioned: mt-4

#### Main Content Grid
- Grid: grid-cols-1 lg:grid-cols-3, gap-6

**Court Diagram Section (2 columns on large screens):** (UPDATED - Conditional Rendering)
- **UPDATED**: Only shows if `match.status === 'completed'`
- If status !== 'completed': Shows message "Processing in progress. Court visualization will appear when processing is complete." (text-gray-400, text-center, py-12)
- If status === 'completed':
  - Dark card (bg-gray-900 or #1a1a1a, rounded-lg, shadow-xl, border border-gray-800, p-6)
  - Title: "Court Visualization" (text-xl, font-semibold, mb-4, text-white)
  - CourtDiagram component:
  - Props: `shots={courtShots}`, `onShotClick={handleShotClick}`
  - `courtShots` is transformed from `shots` array:
    ```typescript
    const courtShots = shots.map((shot) => ({
      id: shot.id,
      start_pos: shot.start_pos,
      end_pos: shot.end_pos,
      result: shot.result,
      video_timestamp: shot.video_timestamp,
    }))
    ```
- Legend (below court):
  - Flex gap-4, text-sm
  - Three items:
    1. Winners: Green line (w-4, h-0.5, bg-green-500) + "Winners" text
    2. Errors: Red dashed line (w-4, h-0.5, border-dashed, border-t-2, border-red-500) + "Errors" text
    3. In Play: Blue line (w-4, h-0.5, bg-blue-500) + "In Play" text

**Video Panel Section (1 column on large screens):**
- Only shows when `showVideoPanel === true && selectedShot !== null`
- Component: `<VideoPanel videoUrl={match.video_url || match.playsight_link} timestamp={selectedShot.video_timestamp || 0} onClose={() => setShowVideoPanel(false)} />`
- Sticky: sticky top-4

#### Stats Section
- Positioned: mt-6
- Component: `<MatchStats match={match} matchData={matchData} shots={shots} />`

#### State Management
- `selectedShot`: any | null (shot object when clicked)
- `showVideoPanel`: boolean (true when shot is clicked)

#### Shot Click Handler
```typescript
const handleShotClick = (shot: any) => {
  setSelectedShot(shot)
  setShowVideoPanel(true)
}
```

---

### 7. UPLOAD MODAL

**Location**: `components/upload/UploadModal.tsx`  
**Props**: `{ isOpen: boolean, onClose: () => void }`

#### Visibility
- If `isOpen === false`: Returns null (not rendered)
- If `isOpen === true`: Renders modal

#### Modal Overlay
- Fixed inset-0, bg-black/80 or bg-black/90, backdrop-blur-sm, flex items-center justify-center, z-50

#### Modal Content
- bg-gray-900 or #1a1a1a, rounded-xl or rounded-2xl, border border-gray-800, shadow-2xl, p-6, max-w-md, w-full, mx-4

#### Header
- Flex justify-between, items-center, mb-4
- Title: "Upload Match Video" (text-xl, font-semibold, text-white)
- Close button: Ghost variant, icon size, X icon from lucide-react, text-gray-400, hover:text-white
  - On click: Calls `onClose()`

#### Form
- Form element with onSubmit handler

**Fields:**

1. **Player Selection (NEW - Coaches Only)**
   - **Only shows if user role === 'coach'**
   - Label: "Select Player" (block, text-sm, font-medium, text-gray-700, mb-2)
   - Dropdown: Select component (shadcn/ui Select)
   - Options: Populated from team members (players only, not coaches)
   - Required: true (for coaches)
   - Value: `selectedPlayerId` (string | null, controlled)
   - On change: Updates `selectedPlayerId`
   - **Data Source**: Fetch team members from teams API
   - **Note**: If coach has no teams/players, show message "Create a team and add players first"

2. **Playsight Link Input**
   - Label: "Playsight Link" (block, text-sm, font-medium, text-gray-700, mb-2)
   - Input: url type, full width, rounded-md, border-gray-300, px-3 py-2, text-sm
   - Placeholder: "https://playsight.com/..."
   - Required: true
   - Value: `playsightLink` (controlled)
   - On change: Updates `playsightLink`

3. **Player Name Input (Optional - Players Only)**
   - **Only shows if user role === 'player'**
   - Label: "Player Name (Optional)" (block, text-sm, font-medium, text-gray-700, mb-2)
   - Input: text type, full width, rounded-md, border-gray-300, px-3 py-2, text-sm
   - Placeholder: "Player name"
   - Value: `playerName` (controlled)
   - On change: Updates `playerName`

**Error Display:**
- If `error` exists:
  - Box: bg-red-50, border-red-200, rounded, p-3
  - Message: text-sm, text-red-800 - shows `error`

**Buttons:**
- Flex gap-2, justify-end
- Cancel button:
  - Text: "Cancel"
  - Variant: outline
  - Type: button
  - Styling: border-gray-700, text-gray-300, hover:border-gray-600, hover:text-white
  - On click: Calls `onClose()`
- Submit button:
  - Text: "Upload" (or "Uploading..." if `loading`)
  - Type: submit
  - Background: emerald green (#50C878)
  - Text: black or white (high contrast)
  - Disabled: when `loading` is true
  - On click: Triggers form submit

#### Form Submit Handler
```typescript
const handleSubmit = async (e: React.FormEvent) => {
  e.preventDefault()
  setError(null)

  // Validation
  if (!playsightLink.trim()) {
    setError('Please enter a Playsight link')
    return
  }
  if (!validatePlaysightLink(playsightLink)) {
    setError('Please enter a valid Playsight link')
    return
  }

  setLoading(true)

  try {
    // Get user and session
    const user = await getUser()
    if (!user) {
      setError('Please sign in first')
      return
    }
    const { data: { session } } = await supabase.auth.getSession()
    if (!session) {
      setError('Please sign in first')
      return
    }

    // Create match via API
    // NEW: For coaches, use selectedPlayerId; for players, use their own user_id
    const matchUserId = userRole === 'coach' && selectedPlayerId 
      ? selectedPlayerId 
      : user.id

    const response = await fetch(`${API_URL}/api/matches`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${session.access_token}`,
      },
      body: JSON.stringify({
        playsight_link: playsightLink,
        player_name: playerName || undefined,
        user_id: matchUserId, // NEW: Pass user_id for match ownership
      }),
    })

    const data = await response.json()

    if (!response.ok) {
      throw new Error(data.detail || 'Failed to create match')
    }

    // Success: Close modal and redirect
    onClose()
    router.push(`/matches/${data.match.id}/identify`)
  } catch (err: any) {
    setError(err.message || 'Failed to upload video')
  } finally {
    setLoading(false)
  }
}
```

#### State
- `selectedPlayerId`: string | null (for coaches, selected player from dropdown)
- `playsightLink`: string
- `playerName`: string (for players only)
- `loading`: boolean
- `error`: string | null
- `userRole`: 'coach' | 'player' (from user profile)

---

### 8. PLAYER IDENTIFICATION PAGE (`/matches/[id]/identify`)

**Route**: `app/matches/[id]/identify/page.tsx`  
**Type**: Protected page (wrapped in MainLayout)  
**Purpose**: User clicks on themselves in video frames

#### Layout
- Background: Black or very dark gray
- Container: mx-auto, px-4, py-8
- Dark card (bg-gray-900 or #1a1a1a, rounded-lg, shadow-xl, border border-gray-800, p-6)

#### Instructions
- Text: "Click on yourself in each frame to help us track your performance throughout the match."
- Style: text-gray-400, mb-6

#### Frames Grid (UPDATED - 5 Frames)
- Grid: grid-cols-1 md:grid-cols-3 lg:grid-cols-5, gap-4, mb-6
- **UPDATED**: Shows 5 frames (instead of 3)
- For each frame in `frames` array (5 frames):
  - Container: relative
  - Frame display:
    - Aspect-video, bg-gray-200, rounded, border-2 border-dashed border-gray-300
    - If placeholder: Shows centered text "Frame {index + 1}" and "Frame extraction coming soon"
    - If real image: Shows image with cursor-crosshair
      - On click: `handleFrameClick(index, event)`
  - Selected indicator:
    - If `selectedCoords[index]` exists:
      - Blue dot: absolute, w-4 h-4, bg-blue-500, rounded-full, border-2 border-white, shadow-lg
      - Position: `left: ${selectedCoords[index].x}%`, `top: ${selectedCoords[index].y}%`
      - Transform: translate(-50%, -50%)
  - Label below: text-xs, text-gray-500, text-center, mt-2
    - Text: "✓ Selected" if coords exist, else "Click to select"

#### Frame Click Handler
```typescript
const handleFrameClick = (frameIndex: number, event: React.MouseEvent<HTMLImageElement>) => {
  const rect = event.currentTarget.getBoundingClientRect()
  const x = ((event.clientX - rect.left) / rect.width) * 100
  const y = ((event.clientY - rect.top) / rect.height) * 100

  const newCoords = [...selectedCoords]
  newCoords[frameIndex] = { x, y }
  setSelectedCoords(newCoords)
}
```

#### Error Display
- If `error` exists:
  - Box: bg-red-900/20, border-red-800, border-2, rounded, p-3, mb-4
  - Message: text-sm, text-red-400 - shows `error`

#### Submit Button
- Position: flex justify-end
- Button: "Submit & Start Processing" (or "Submitting..." if `submitting`)
- Disabled: when `submitting === true || selectedCoords.length === 0`
- On click: `handleSubmit` function:
  - Validates: If `selectedCoords.length === 0`, sets error and returns
  - Sets `submitting` to true, clears error
  - Gets user and session
  - Calls API: `POST /api/videos/identify-player`
    - Body: `{ match_id, frame_data: { frames }, selected_player_coords }`
  - On success: Redirects to `/matches/${matchId}`
  - On error: Sets error message

#### State
- `frames`: string[] (placeholder frames initially)
- `selectedCoords`: Array<{ x: number, y: number }>
- `loading`: boolean
- `submitting`: boolean
- `error`: string | null

---

### 9. STATS PAGE (`/stats`) (UPDATED - Summary + Charts)

**Route**: `app/stats/page.tsx`  
**Type**: Protected page (wrapped in MainLayout)  
**Purpose**: Display player/coach statistics with summary cards and detailed charts

#### Layout
- Container: mx-auto, px-4, py-8
- Title: "Statistics" (text-3xl, font-bold, mb-6)

#### Coach View (when `profile.role === 'coach'`) (UPDATED)
- **Player Selection** (at top):
  - Label: "Select Player:" (text-sm, font-medium, text-gray-700, mr-2)
  - Dropdown: Select component
  - Options: "All Players" + list of team member names
  - State: `selectedPlayerId` (string | null)
  - On change: Updates stats display for selected player

- **Summary Cards Section** (NEW):
  - Grid: grid-cols-1 md:grid-cols-2 lg:grid-cols-4, gap-4, mb-6
  - Cards display:
    1. **Total Matches**: Number of completed matches
    2. **Total Shots**: Sum of all shots across matches
    3. **Win Rate**: Percentage (if applicable) or winners vs errors
    4. **Average Shots per Match**: Total shots / total matches
  - Each card: bg-gray-900 or #1a1a1a, rounded-lg, shadow-xl, border border-gray-800, p-6
  - Card layout:
    - Title: text-sm, font-medium, text-gray-400, mb-2
    - Value: text-3xl, font-bold, text-white
    - Optional: Trend indicator or comparison (emerald green accent)

- **Detailed Charts Section** (NEW):
  - Grid: grid-cols-1 lg:grid-cols-2, gap-6
  - Charts (use Recharts library or similar):
    1. **Winners/Errors Over Time**: Line chart showing trends across matches
    2. **Shot Distribution**: Pie/bar chart showing winners vs errors vs in-play
    3. **Performance by Match**: Bar chart showing stats per match
    4. **Shot Patterns**: Heatmap or scatter plot of shot locations
  - Each chart: bg-gray-900 or #1a1a1a, rounded-lg, shadow-xl, border border-gray-800, p-6
  - Chart title: text-lg, font-semibold, mb-4, text-white
  - Chart colors: Use emerald green (#50C878) as primary color, with variations for different data series

#### Player View (when `profile.role === 'player'`) (UPDATED)
- **Summary Cards Section** (NEW):
  - Same layout as coach view (4 summary cards)
  - Shows player's own stats only

- **Detailed Charts Section** (NEW):
  - Same charts as coach view
  - Shows player's own data only

#### Empty State
- If no matches/stats available:
  - Message: "No statistics yet. Upload matches to see your performance analytics." (text-gray-600, text-center, py-12)

---

### 10. PROFILE PAGE (NEW)

**Route**: `app/profile/page.tsx`  
**Type**: Protected page (wrapped in MainLayout)  
**Purpose**: Display user profile information

#### Layout
- Container: mx-auto, px-4, py-8
- Title: "Profile" (text-3xl, font-bold, mb-6)

#### Profile Card
- White card (bg-white, rounded-lg, shadow, p-6)
- Grid layout: grid-cols-1 md:grid-cols-2, gap-6

#### Profile Information (View-Only)
1. **Name**
   - Label: "Name" (text-sm, font-medium, text-gray-400, mb-1)
   - Value: `profile.name || 'Not set'` (text-base, text-white)

2. **Email**
   - Label: "Email" (text-sm, font-medium, text-gray-400, mb-1)
   - Value: `user.email` (text-base, text-white)

3. **Role**
   - Label: "Role" (text-sm, font-medium, text-gray-400, mb-1)
   - Value: Badge showing `profile.role` (capitalized: "Coach" or "Player")
   - Badge styling:
     - Coach: bg-blue-900/30, text-blue-400, border border-blue-800
     - Player: bg-emerald-900/30, text-emerald-400, border border-emerald-800

4. **Teams Section**
   - Label: "Teams" (text-sm, font-medium, text-gray-400, mb-2)
   - If teams.length === 0:
     - Message: "Not part of any teams yet." (text-sm, text-gray-500)
   - If teams.length > 0:
     - List: space-y-2
     - Each team: Card (bg-black/50, rounded, border border-gray-800, p-3)
       - Team name: font-medium, text-white
       - Team code: text-xs, text-gray-400, font-mono

#### Empty State
- If no teams:
  - Helper message: "Join a team to get started!" (text-sm, text-gray-500, italic)

#### Future (Not Implemented Yet)
- Edit button (disabled/hidden for now)
- Profile picture upload
- Additional fields (phone, bio, etc.)

---

### 11. SIDEBAR COMPONENT (UPDATED - Logo + Profile Icon)

**Location**: `components/layout/Sidebar.tsx`

#### Layout
- Fixed left sidebar: w-64, h-screen, flex-col
- Background: Black (#000000) or very dark gray (#0a0a0a)
- Text: White
- Border-right: border-gray-800 (subtle border)

#### Header (UPDATED - Logo + Brand)
- Height: h-16 or h-20, flex items-center, px-4, border-b border-gray-800
- **Logo + Brand Name**:
  - Flex items-center gap-3
  - Logo image: Minimalist owl's eyes drawing (use Next.js Image component: `import Image from 'next/image'`)
  - Logo path: `/logo.png` or `/logo.svg` (in public folder)
  - Logo size: w-10 h-10 or w-12 h-12
  - Brand name: "Courtvision" (text-lg or text-xl, font-bold, text-white)
  - Optional: Link to dashboard (Next.js Link, hover effect)

#### Profile Icon (NEW - Below Logo)
- Position: Absolute bottom or in footer section
- Icon: User icon from lucide-react (User, UserCircle, or similar)
- Size: w-8, h-8
- Styling: text-gray-300, hover:text-emerald-green, cursor-pointer
- Position: Can be in header (top-right) or footer (bottom)
- On click: Opens profile dropdown menu

#### Navigation
- Flex-1, space-y-1, px-2, py-4
- Navigation items:
  ```typescript
  [
    { name: 'Dashboard', href: '/dashboard', icon: Home }, // Use lucide-react icons
    { name: 'Stats', href: '/stats', icon: BarChart3 },
    { name: 'Teams', href: '/teams', icon: Users },
  ]
  ```
- Each item:
  - Link component (Next.js Link)
  - Flex items-center gap-3, rounded-lg, px-3, py-2, text-sm, font-medium
  - Active state: bg-emerald-green/20, text-emerald-green, border-l-2 border-emerald-green (when `pathname === item.href`)
  - Inactive state: text-gray-300, hover:bg-gray-800/50, hover:text-white
  - Icon: lucide-react icon, size w-5 h-5
  - Text: item.name

#### Footer
- Border-top: border-gray-800, p-4
- **Profile Icon + Dropdown**:
  - Profile icon button: Full width or centered
  - Styling: text-gray-300, hover:text-emerald-green, cursor-pointer
  - On click: Opens profile dropdown menu

#### Profile Dropdown Menu (NEW)
- **Trigger**: Profile icon in footer or header
- **Component**: Use shadcn/ui DropdownMenu or Popover
- **Position**: Opens above profile icon (if in footer) or below (if in header)
- **Styling**: 
  - Background: Dark gray (#262626) or black
  - Border: border-gray-800
  - Shadow: shadow-xl
  - Rounded: rounded-lg
  - Padding: py-2
- **Menu Items**:
  1. **Profile** (with User icon from lucide-react)
     - Styling: px-4, py-2, hover:bg-gray-800, text-white, flex items-center gap-2
     - On click: Navigate to `/profile` using Next.js router, close dropdown
  2. **Divider** (separator line, border-gray-800)
  3. **Sign Out** (with LogOut icon from lucide-react)
     - Styling: px-4, py-2, hover:bg-gray-800, text-white, flex items-center gap-2
     - On click: `handleSignOut`:
       - Sets `isSigningOut` to true
       - Calls `signOut()` from useAuth
       - Sets `isSigningOut` to false
       - Closes dropdown

#### State
- `isSigningOut`: boolean
- `isProfileMenuOpen`: boolean (controls dropdown visibility)

---

### 12. FLOATING ACTION BUTTON

**Location**: `components/layout/FloatingActionButton.tsx`

#### Position
- Fixed: bottom-6, right-6, z-40

#### Button
- Circular: h-14, w-14, rounded-full
- Background: emerald green (#50C878)
- Shadow: shadow-lg, shadow-emerald-500/50
- Icon: Plus from lucide-react (h-6, w-6, text-black or white for contrast)
- Hover: Slight scale (scale-110) or brighter shade
- On click: Sets `isModalOpen` to true

#### Modal
- Component: `<UploadModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} />`

#### State
- `isModalOpen`: boolean

---

### 13. PROCESSING STATUS COMPONENT

**Location**: `components/upload/ProcessingStatus.tsx`  
**Props**: `{ matchId: string }`

#### Purpose
- Shows real-time processing status for a match
- Polls every 5 seconds if status is "processing"
- Subscribes to Supabase real-time updates

#### Layout
- Dark card: bg-gray-900 or #1a1a1a, rounded-lg, shadow-xl, border border-gray-800, p-6
- Flex items-center gap-4

#### Status Badge
- px-4, py-2, rounded-lg
- Colors based on status (on dark background):
  - pending: bg-amber-900/30, text-amber-400, border border-amber-800
  - processing: bg-blue-900/30, text-blue-400, border border-blue-800
  - completed: bg-emerald-900/30, text-emerald-400, border border-emerald-800
  - failed: bg-red-900/30, text-red-400, border border-red-800
- Text: font-semibold, status label

#### Message
- flex-1
- Message text: text-sm, text-gray-400
- Messages:
  - pending: "Waiting to start processing..."
  - processing: "Analyzing video... This may take up to an hour."
  - completed: "Processing complete! Your match is ready."
  - failed: "Processing failed. Please try again."
- If `processed_at` exists: Shows completion time (text-xs, text-gray-500, mt-1)

#### Loading Spinner
- Only shows when status === "processing"
- Animate-spin, rounded-full, h-6, w-6, border-b-2, border-blue-600

---

### 14. COURT DIAGRAM COMPONENT

**Location**: `components/court/CourtDiagram.tsx`  
**Props**: `{ shots?: Shot[], onShotClick?: (shot: Shot) => void }`

#### Layout
- Container: w-full, bg-gray-900 or #1a1a1a, rounded-lg, border border-gray-800, p-4, overflow-x-auto
- SVG: width 800, height 400, viewBox, mx-auto

#### Court Elements
- Court outline: Green rectangle (fill #4ade80, stroke #22c55e, strokeWidth 2)
- Net line: White vertical line at center (stroke white, strokeWidth 3)
- Service boxes: 4 rectangles (lighter green #86efac)
- Center service line: White horizontal line
- Service box dividers: White vertical lines

#### Shots
- Maps through `shots` array
- Renders `ShotLine` component for each shot
- Passes: `shot`, `courtWidth`, `courtHeight`, `onClick` handler

---

### 15. VIDEO PANEL COMPONENT

**Location**: `components/video/VideoPanel.tsx`  
**Props**: `{ videoUrl: string, timestamp: number, onClose: () => void }`

#### Layout
- Dark card: bg-gray-900 or #1a1a1a, rounded-lg, shadow-xl, border border-gray-800, p-4, sticky top-4

#### Header
- Flex justify-between, items-center, mb-4
- Title: "Video" (text-lg, font-semibold, text-white)
- Close button: Ghost variant, icon size, X icon, text-gray-400, hover:text-white
  - On click: Calls `onClose()`

#### Video Container
- aspect-video, bg-black, rounded, overflow-hidden

#### Video Display
- If Playsight link: iframe with `src={embedUrl}`, full width/height, allowFullScreen
- If regular video: video element with controls, sets `currentTime` to `timestamp` on load

#### Timestamp Display
- If `timestamp > 0`: Shows "Jump to: MM:SS" (text-sm, text-gray-600, mt-2)

---

## API INTEGRATION PATTERNS

### All API Calls Require Authentication
```typescript
const { getSession } = useAuth()
const session = await getSession()
const token = session?.access_token

const response = await fetch(`${API_URL}/api/endpoint`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`,
  },
  body: JSON.stringify(data),
})
```

### API Base URL
```typescript
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
```

### Error Handling
- Always check `response.ok`
- Parse JSON: `const data = await response.json()`
- Throw error with `data.detail` if available
- Display error to user in UI

---

## COLOR SCHEME

### Primary Colors
- **Background**: Black (#000000 or very dark gray #0a0a0a)
- **Primary Accent**: Emerald Green (#50C878)
- **Highlights**: White (#FFFFFF)
- **Text on Dark**: White or light gray (#F5F5F5)
- **Text on Light**: Black or dark gray (#1a1a1a)

### Usage Guidelines
- **Main Background**: Black/dark theme throughout
- **Primary Buttons**: Emerald green (#50C878) background, white text
- **Secondary Buttons**: White background, emerald green text, or outlined
- **Accent Elements**: Emerald green for highlights, CTAs, active states
- **Cards/Containers**: Dark gray (#1a1a1a or #262626) on black background
- **Borders**: Subtle gray (#333333) or emerald green for emphasis

### Color Variables (Add to globals.css)
```css
--background: #000000;
--foreground: #FFFFFF;
--primary: #50C878;
--primary-foreground: #000000;
--secondary: #1a1a1a;
--secondary-foreground: #FFFFFF;
--muted: #262626;
--muted-foreground: #a0a0a0;
--accent: #50C878;
--accent-foreground: #000000;
--border: #333333;
--input: #1a1a1a;
--ring: #50C878;
```

### Status Colors (on dark background)
- Success: Emerald green (#50C878)
- Error: Red (#ef4444)
- Warning: Amber (#f59e0b)
- Info: Emerald green (#50C878)

### Spacing
- Container padding: px-4, py-8
- Card padding: p-6
- Gap between elements: gap-4, gap-6
- Section margins: mb-4, mb-6

### Typography
- Page titles: text-3xl, font-bold
- Section titles: text-xl, font-semibold
- Card titles: text-lg, font-semibold
- Body text: text-sm, text-gray-600
- Muted text: text-xs, text-gray-500

### Components
- Use shadcn/ui Button component
- Use lucide-react icons (Home, BarChart3, Users, LogOut, Plus, X, User, etc.)
- All inputs: 
  - Dark theme: bg-black/50, border-gray-700, text-white, placeholder-gray-500
  - Focus: border-emerald-green, ring-emerald-green
- All cards: 
  - Dark theme: bg-gray-900 or #1a1a1a, border-gray-800
  - Rounded: rounded-lg or rounded-xl
  - Shadow: shadow-xl (on dark background)

---

## IMPORTANT NOTES

1. **DO NOT** create new hooks - use existing `useAuth`, `useMatches`, `useTeams`
2. **DO NOT** create new API clients - use fetch with Bearer tokens
3. **DO** use existing TypeScript types
4. **DO** use `'use client'` directive for all client components
5. **DO** handle loading and error states
6. **DO** use TanStack Query via existing hooks (not directly)
7. **DO** wrap protected pages in MainLayout
8. **DO** use Next.js Link for navigation
9. **DO** use Next.js router for programmatic navigation
10. **DO** handle form submissions with preventDefault
11. **DO** validate inputs before API calls
12. **DO** show loading states during async operations
13. **DO** display error messages to users
14. **DO** use controlled inputs (value + onChange)
15. **DO** use conditional rendering for empty/loading/error states
16. **NEW**: For player dropdown in upload modal, fetch team members from `/api/teams/{team_id}/members` endpoint
17. **NEW**: For player filter on dashboard, fetch team members list (can use same endpoint or teams API)
18. **NEW**: Update `useAuth` hook to accept `role` parameter in `signUp` function
19. **NEW**: Profile page should fetch user data and teams using existing hooks/APIs
20. **NEW**: Match detail page should only show court diagram when `match.status === 'completed'`
21. **NEW**: Landing page is the entry point - beautiful hero section, features, testimonials
22. **NEW**: Auth happens via modal on landing page (not separate route)
23. **NEW**: Logo appears in top left of landing page nav AND top of sidebar in dashboard
24. **NEW**: Use dark theme (black) with emerald green (#50C878) accents throughout
25. **NEW**: All cards, modals, and components use dark gray backgrounds on black
26. **NEW**: Primary buttons use emerald green (#50C878), secondary use outline with white/gray
27. **NEW**: Landing page should be modern and impressive - hero video/images, smooth scrolling, professional design

---

## COMPLETE USER FLOWS

### Flow 1: Landing Page → Sign Up → Dashboard (UPDATED)
1. User visits `/` (landing page)
2. Sees beautiful hero section with "Get Started" or "Sign Up" button
3. Clicks "Sign Up" button (top right or hero CTA)
4. **NEW**: Auth modal opens
5. **NEW**: Sees role selection (Coach/Player radio buttons)
6. Selects role (defaults to Player)
7. Fills in name, email, password
8. Clicks "Sign up"
9. Modal closes
10. If email confirmations disabled: Auto-redirects to `/dashboard`
11. If email confirmations enabled: Shows alert, user confirms email, then signs in
12. **NEW**: User role is set in database based on selection

### Flow 2: Create Team (Coach)
1. Coach goes to `/teams`
2. Clicks "Create New Team"
3. Enters team name
4. Clicks "Create Team"
5. Sees success message with team code
6. Can create another team or see team in list

### Flow 3: Join Team (Player)
1. Player goes to `/teams`
2. Enters team code (auto-uppercased)
3. Clicks "Join"
4. On success: Team appears in "Your Teams" list
5. On error: Shows error message

### Flow 4: Upload Match (UPDATED - Coach Upload for Player)
1. User clicks floating "+" button
2. Modal opens
3. **NEW (Coaches)**: Sees player dropdown, selects player from team members
4. **NEW (Players)**: Sees optional player name field (or no field if uploading for self)
5. Enters Playsight link
6. Clicks "Upload"
7. **NEW**: Match is created with `user_id` = selected player's ID (for coaches) or own ID (for players)
8. Modal closes
9. Redirects to `/matches/[id]/identify`
10. **NEW**: Match appears in BOTH coach's and selected player's dashboard

### Flow 5: Player Identification (UPDATED - 5 Frames)
1. User sees **5 frames** (placeholder for now, updated from 3)
2. Clicks on themselves in each frame
3. Blue dot appears at click location
4. Clicks "Submit & Start Processing"
5. Redirects to `/matches/[id]` (match detail page)

### Flow 6: View Match (UPDATED - Wait for Completion)
1. User clicks match card from dashboard
2. Goes to `/matches/[id]`
3. **UPDATED**: If status !== 'completed': Shows processing status only (no court diagram)
4. **UPDATED**: If status === 'completed': Shows court diagram with shots
5. Clicks on shot line → Video panel opens
6. Video panel shows Playsight embed at shot timestamp
7. Can close video panel
8. Sees match stats below court

### Flow 7: Dashboard with Player Filter (NEW - Coaches)
1. Coach visits `/dashboard`
2. **NEW**: Sees player filter dropdown at top (below title)
3. Default: "All Players" selected (shows all team matches)
4. Selects specific player from dropdown
5. Dashboard updates to show only that player's matches
6. Still grouped by date
7. Can switch back to "All Players" to see all matches

### Flow 8: Profile Access (NEW)
1. User clicks profile icon in sidebar header
2. Dropdown menu appears
3. Options: "Profile" or "Sign Out"
4. Clicks "Profile" → Navigates to `/profile`
5. Sees profile information (name, email, role, teams)
6. Can click "Sign Out" from dropdown to sign out

---

**END OF SPECIFICATION**

Copy this entire document into v0 when generating components. Be extremely detailed about every interaction, button click, state change, and user flow.
