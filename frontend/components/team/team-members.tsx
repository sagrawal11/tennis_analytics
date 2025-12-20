'use client'

import { useTeamMembers } from '@/hooks/useTeamMembers'

interface TeamMembersProps {
  teamId: string
}

export function TeamMembers({ teamId }: TeamMembersProps) {
  const { data: members, isLoading } = useTeamMembers(teamId)

  if (isLoading) {
    return <div className="text-sm text-gray-400">Loading members...</div>
  }

  if (!members || members.length === 0) {
    return <div className="text-sm text-gray-400">No team members yet.</div>
  }

  return (
    <div className="space-y-2">
      {members.map((member: any) => (
        <div key={member.id} className="flex items-center justify-between p-3 bg-[#1a1a1a] rounded-lg border border-[#333333]">
          <div>
            <p className="font-medium text-white">{member.users?.name || member.users?.email || 'Unknown'}</p>
            <p className="text-sm text-gray-400 capitalize">{member.role}</p>
          </div>
        </div>
      ))}
    </div>
  )
}
