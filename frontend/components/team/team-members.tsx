'use client'

import { useTeamMembers } from '@/hooks/useTeamMembers'
import { useAuth } from '@/hooks/useAuth'
import { useProfile } from '@/hooks/useProfile'
import { useEffect, useState } from 'react'
import { Button } from '@/components/ui/button'
import { X } from 'lucide-react'
import { RemovePlayerModal } from './remove-player-modal'

interface TeamMembersProps {
  teamId: string
  teamName: string
}

export function TeamMembers({ teamId, teamName }: TeamMembersProps) {
  const { data: members, isLoading, removeMember, isRemoving } = useTeamMembers(teamId)
  const { getUser } = useAuth()
  const { profile } = useProfile()
  const [currentUserId, setCurrentUserId] = useState<string | null>(null)
  const [removeModalOpen, setRemoveModalOpen] = useState(false)
  const [selectedPlayerId, setSelectedPlayerId] = useState<string | null>(null)
  const [selectedPlayerName, setSelectedPlayerName] = useState<string | null>(null)

  const isCoach = profile?.role === 'coach'

  useEffect(() => {
    const fetchUserId = async () => {
      const user = await getUser()
      setCurrentUserId(user?.id || null)
    }
    fetchUserId()
  }, [getUser])

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
            {coaches.map((member: any) => {
              const isCurrentUser = member.users?.id === currentUserId
              return (
                <div key={member.id} className={`flex items-center justify-between p-3 bg-black/50 rounded-lg border ${isCurrentUser ? 'border-[#50C878]' : 'border-[#333333]'}`}>
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
              )
            })}
          </div>
        </div>
      )}

      {players.length > 0 && (
        <div>
          <h5 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Players</h5>
          <div className="space-y-2">
            {players.map((member: any) => {
              const isCurrentUser = member.users?.id === currentUserId
              const playerName = member.users?.name || member.users?.email || 'Unknown'
              return (
                <div key={member.id} className={`flex items-center justify-between p-3 bg-black/50 rounded-lg border transition-all duration-200 ease-in-out ${isCurrentUser ? 'border-[#50C878]' : 'border-[#333333] hover:border-[#50C878]/50'}`}>
                  <div className="flex items-center gap-3">
                    <div>
                      <p className="font-medium text-white transition-colors duration-200">{playerName}</p>
                      <p className="text-xs text-gray-500 transition-colors duration-200">{member.users?.email}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="bg-emerald-900/30 text-emerald-400 border border-emerald-800 rounded px-2 py-1 text-xs font-medium transition-all duration-200">
                      Player
                    </span>
                    {isCoach && !isCurrentUser && (
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 text-red-400 hover:text-red-300 hover:bg-red-900/20 transition-all duration-200 ease-in-out"
                        onClick={() => {
                          setSelectedPlayerId(member.users?.id)
                          setSelectedPlayerName(playerName)
                          setRemoveModalOpen(true)
                        }}
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {selectedPlayerId && selectedPlayerName && (
        <RemovePlayerModal
          isOpen={removeModalOpen}
          onClose={() => {
            setRemoveModalOpen(false)
            setSelectedPlayerId(null)
            setSelectedPlayerName(null)
          }}
          onConfirm={async () => {
            if (selectedPlayerId) {
              try {
                await removeMember(selectedPlayerId)
                setRemoveModalOpen(false)
                setSelectedPlayerId(null)
                setSelectedPlayerName(null)
              } catch (error) {
                console.error('Failed to remove player:', error)
              }
            }
          }}
          playerName={selectedPlayerName}
          teamName={teamName}
          loading={isRemoving}
        />
      )}
    </div>
  )
}
