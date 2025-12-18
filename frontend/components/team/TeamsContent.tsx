'use client'

import { useState } from 'react'
import { CreateTeam } from './CreateTeam'
import { TeamCode } from './TeamCode'
import { TeamMembers } from './TeamMembers'
import { useTeams } from '@/hooks/useTeams'
import { Button } from '@/components/ui/button'

interface TeamsContentProps {
  profile: {
    role?: string
  } | null
}

export function TeamsContent({ profile }: TeamsContentProps) {
  const [showCreate, setShowCreate] = useState(false)
  const isCoach = profile?.role === 'coach'
  const { teams, isLoading } = useTeams()

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-6">Teams</h1>

      {isCoach ? (
        <div className="space-y-8">
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold mb-4">Create Team</h2>
            {showCreate ? (
              <CreateTeam
                onCreated={() => {
                  setShowCreate(false)
                }}
              />
            ) : (
              <Button onClick={() => setShowCreate(true)}>Create New Team</Button>
            )}
          </div>

          {!isLoading && teams && teams.length > 0 && (
            <div className="space-y-4">
              {teams.map((team: any) => (
                <div key={team.id} className="bg-white rounded-lg shadow p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold">{team.name}</h3>
                    <div className="bg-gray-100 rounded px-3 py-1">
                      <span className="text-sm font-mono font-semibold">{team.code}</span>
                    </div>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-gray-700 mb-2">Team Members</h4>
                    <TeamMembers teamId={team.id} />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      ) : (
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold mb-4">Join Team</h2>
          <TeamCode onJoin={() => {}} />
          
          {!isLoading && teams && teams.length > 0 && (
            <div className="mt-8">
              <h3 className="text-lg font-semibold mb-4">Your Teams</h3>
              <div className="space-y-2">
                {teams.map((team: any) => (
                  <div key={team.id} className="p-3 bg-gray-50 rounded-lg">
                    <p className="font-medium">{team.name}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
