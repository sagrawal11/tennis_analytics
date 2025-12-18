'use client'

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { createClient } from '@/lib/supabase/client'
import { useAuth } from './useAuth'

export function useTeams() {
  const { getUser } = useAuth()
  const supabase = createClient()
  const queryClient = useQueryClient()

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

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/teams/join`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${session.access_token}`,
        },
        body: JSON.stringify({ code }),
      })

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || 'Failed to join team')
      }

      return response.json()
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['my-teams'] })
    },
  })

  return {
    teams: teams || [],
    isLoading,
    createTeam: createTeamMutation.mutateAsync,
    joinTeam: joinTeamMutation.mutateAsync,
    isCreating: createTeamMutation.isPending,
    isJoining: joinTeamMutation.isPending,
  }
}
