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

  // Separate coaches and players
  const coaches = members.filter((m: any) => m.role === 'coach')
  const players = members.filter((m: any) => m.role === 'player')

  return (
    <div className="space-y-4">
      {coaches.length > 0 && (
        <div>
          <h5 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Coaches</h5>
          <div className="space-y-2">
            {coaches.map((member: any) => (
              <div key={member.id} className="flex items-center justify-between p-3 bg-black/50 rounded-lg border border-[#333333]">
                <div className="flex items-center gap-3">
                  <div>
                    <p className="font-medium text-white">{member.users?.name || member.users?.email || 'Unknown'}</p>
                    <p className="text-xs text-gray-500">{member.users?.email}</p>
                  </div>
                </div>
                <span className="bg-blue-900/30 text-blue-400 border border-blue-800 rounded px-2 py-1 text-xs font-medium">
                  Coach
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {players.length > 0 && (
        <div>
          <h5 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Players</h5>
          <div className="space-y-2">
            {players.map((member: any) => (
              <div key={member.id} className="flex items-center justify-between p-3 bg-black/50 rounded-lg border border-[#333333]">
                <div className="flex items-center gap-3">
                  <div>
                    <p className="font-medium text-white">{member.users?.name || member.users?.email || 'Unknown'}</p>
                    <p className="text-xs text-gray-500">{member.users?.email}</p>
                  </div>
                </div>
                <span className="bg-emerald-900/30 text-emerald-400 border border-emerald-800 rounded px-2 py-1 text-xs font-medium">
                  Player
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
