'use client'

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { createClient } from '@/lib/supabase/client'
import { useAuth } from './useAuth'

export function useTeamMembers(teamId: string) {
  const { getUser } = useAuth()
  const supabase = createClient()
  const queryClient = useQueryClient()

  const query = useQuery({
    queryKey: ['team-members', teamId],
    queryFn: async () => {
      const user = await getUser()
      if (!user) return []

      const { data: { session } } = await supabase.auth.getSession()
      if (!session) return []

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/teams/${teamId}/members`, {
        headers: {
          'Authorization': `Bearer ${session.access_token}`,
        },
      })

      if (!response.ok) return []

      const data = await response.json()
      return data.members || []
    },
  })

  const removeMemberMutation = useMutation({
    mutationFn: async (userId: string) => {
      const user = await getUser()
      if (!user) throw new Error('Not authenticated')

      const { data: { session } } = await supabase.auth.getSession()
      if (!session) throw new Error('Not authenticated')

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/teams/${teamId}/members/${userId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${session.access_token}`,
        },
      })

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || 'Failed to remove team member')
      }

      return response.json()
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['team-members', teamId] })
      queryClient.invalidateQueries({ queryKey: ['my-teams'] })
    },
  })

  return {
    ...query,
    removeMember: removeMemberMutation.mutateAsync,
    isRemoving: removeMemberMutation.isPending,
  }
}
