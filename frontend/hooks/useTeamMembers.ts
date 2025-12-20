'use client'

import { useQuery } from '@tanstack/react-query'
import { createClient } from '@/lib/supabase/client'
import { useAuth } from './useAuth'

export function useTeamMembers(teamId: string) {
  const { getUser } = useAuth()
  const supabase = createClient()

  return useQuery({
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
}
