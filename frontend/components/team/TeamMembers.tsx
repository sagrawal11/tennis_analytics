'use client'

import { useQuery } from '@tanstack/react-query'
import { createClient } from '@/lib/supabase/client'
import { useAuth } from '@/hooks/useAuth'

interface TeamMembersProps {
  teamId: string
}

export function TeamMembers({ teamId }: TeamMembersProps) {
  const { getUser } = useAuth()
  const supabase = createClient()

  const { data: members, isLoading } = useQuery({
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

  if (isLoading) {
    return <div className="text-sm text-gray-600">Loading members...</div>
  }

  if (!members || members.length === 0) {
    return <div className="text-sm text-gray-600">No team members yet.</div>
  }

  return (
    <div className="space-y-2">
      {members.map((member: any) => (
        <div key={member.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
          <div>
            <p className="font-medium">{member.users?.name || member.users?.email || 'Unknown'}</p>
            <p className="text-sm text-gray-600 capitalize">{member.role}</p>
          </div>
        </div>
      ))}
    </div>
  )
}
