'use client'

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { createClient } from '@/lib/supabase/client'
import { useAuth } from './useAuth'

export function useTeams() {
  const { getUser } = useAuth()
  const supabase = createClient()
  const queryClient = useQueryClient()
  
  // Note: AbortSignal.timeout might not be available in all browsers
  // Fallback to manual timeout if needed

  const { data: teams, isLoading } = useQuery({
    queryKey: ['my-teams'],
    queryFn: async () => {
      const user = await getUser()
      if (!user) return []

      const { data: { session } } = await supabase.auth.getSession()
      if (!session) return []

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/teams/my-teams`, {
        headers: {
          'Authorization': `Bearer ${session.access_token}`,
        },
      })

      if (!response.ok) return []

      const data = await response.json()
      return data.teams || []
    },
  })

  const createTeamMutation = useMutation({
    mutationFn: async (name: string) => {
      const user = await getUser()
      if (!user) throw new Error('Not authenticated')

      const { data: { session } } = await supabase.auth.getSession()
      if (!session) throw new Error('Not authenticated')

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/teams/create`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${session.access_token}`,
        },
        body: JSON.stringify({ name }),
      })

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || 'Failed to create team')
      }

      return response.json()
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['my-teams'] })
    },
  })

  const joinTeamMutation = useMutation({
    mutationFn: async (code: string) => {
      const user = await getUser()
      if (!user) throw new Error('Not authenticated')

      const { data: { session } } = await supabase.auth.getSession()
      if (!session) throw new Error('Not authenticated')

      let response: Response
      try {
        response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/teams/join`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${session.access_token}`,
          },
          body: JSON.stringify({ code }),
        })
      } catch (networkError: any) {
        // Network error (CORS, connection refused, timeout, etc.)
        console.error('Network error joining team:', networkError)
        
        // If it's an AbortError (timeout), the request might have still succeeded
        if (networkError.name === 'AbortError' || networkError.name === 'TimeoutError') {
          // Request might have succeeded - invalidate queries to check
          queryClient.invalidateQueries({ queryKey: ['my-teams'] })
          throw new Error('Request timed out. Please refresh the page to check if you joined successfully.')
        }
        
        throw new Error('Network error: Could not connect to server. Please check your connection and try again.')
      }

      // Try to parse response, but handle errors gracefully
      let data: any
      try {
        data = await response.json()
      } catch (parseError) {
        // Response is not JSON - might be empty or error
        if (!response.ok) {
          throw new Error(`Server error: ${response.status} ${response.statusText}`)
        }
        // Response is ok but not JSON - assume success
        return { message: 'Successfully joined team', team: null }
      }

      if (!response.ok) {
        throw new Error(data.detail || data.message || `Failed to join team: ${response.status}`)
      }

      return data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['my-teams'] })
      queryClient.invalidateQueries({ queryKey: ['team-activation-status'] })
      queryClient.invalidateQueries({ queryKey: ['activation-status'] })
      queryClient.invalidateQueries({ queryKey: ['profile'] })
    },
  })

  const archiveTeamMutation = useMutation({
    mutationFn: async (teamId: string) => {
      const user = await getUser()
      if (!user) throw new Error('Not authenticated')

      const { data: { session } } = await supabase.auth.getSession()
      if (!session) throw new Error('Not authenticated')

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/teams/${teamId}/archive`, {
        method: 'PATCH',
        headers: {
          'Authorization': `Bearer ${session.access_token}`,
        },
      })

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || 'Failed to archive team')
      }

      return response.json()
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['my-teams'] })
      queryClient.invalidateQueries({ queryKey: ['archived-teams'] })
    },
  })

  const unarchiveTeamMutation = useMutation({
    mutationFn: async (teamId: string) => {
      const user = await getUser()
      if (!user) throw new Error('Not authenticated')

      const { data: { session } } = await supabase.auth.getSession()
      if (!session) throw new Error('Not authenticated')

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/teams/${teamId}/unarchive`, {
        method: 'PATCH',
        headers: {
          'Authorization': `Bearer ${session.access_token}`,
        },
      })

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || 'Failed to unarchive team')
      }

      return response.json()
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['my-teams'] })
      queryClient.invalidateQueries({ queryKey: ['archived-teams'] })
    },
  })

  // Query for archived teams (coaches only)
  const { data: archivedTeams } = useQuery({
    queryKey: ['archived-teams'],
    queryFn: async () => {
      const user = await getUser()
      if (!user) return []

      const { data: { session } } = await supabase.auth.getSession()
      if (!session) return []

      // Check if user is a coach
      const profileResponse = await supabase.from('users').select('role').eq('id', user.id).single()
      if (!profileResponse.data || profileResponse.data.role !== 'coach') {
        return []
      }

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/teams/my-teams?include_archived=true`, {
        headers: {
          'Authorization': `Bearer ${session.access_token}`,
        },
      })

      if (!response.ok) return []

      const data = await response.json()
      // Filter to only archived teams
      return (data.teams || []).filter((team: any) => team.status === 'archived')
    },
  })

  return {
    teams: teams || [],
    archivedTeams: archivedTeams || [],
    isLoading,
    createTeam: createTeamMutation.mutateAsync,
    joinTeam: joinTeamMutation.mutateAsync,
    archiveTeam: archiveTeamMutation.mutateAsync,
    unarchiveTeam: unarchiveTeamMutation.mutateAsync,
    isCreating: createTeamMutation.isPending,
    isJoining: joinTeamMutation.isPending,
    isArchiving: archiveTeamMutation.isPending,
    isUnarchiving: unarchiveTeamMutation.isPending,
  }
}
