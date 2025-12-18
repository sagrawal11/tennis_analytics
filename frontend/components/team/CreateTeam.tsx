'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { useTeams } from '@/hooks/useTeams'

interface CreateTeamProps {
  onCreated?: () => void
}

export function CreateTeam({ onCreated }: CreateTeamProps) {
  const [teamName, setTeamName] = useState('')
  const [teamCode, setTeamCode] = useState<string | null>(null)
  const { createTeam, isCreating } = useTeams()
  const [error, setError] = useState<string | null>(null)

  const handleCreate = async () => {
    if (!teamName.trim()) {
      setError('Please enter a team name')
      return
    }

    setError(null)

    try {
      const data = await createTeam(teamName)
      setTeamCode(data.code)
      setTeamName('')
      onCreated?.()
    } catch (err: any) {
      setError(err.message || 'Failed to create team')
    }
  }

  if (teamCode) {
    return (
      <div className="space-y-4">
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <p className="text-sm font-medium text-green-800 mb-2">Team created successfully!</p>
          <p className="text-sm text-green-700 mb-4">Share this code with your players:</p>
          <div className="bg-white rounded border-2 border-green-500 p-4 text-center">
            <p className="text-3xl font-bold text-green-700">{teamCode}</p>
          </div>
        </div>
        <Button
          onClick={() => {
            setTeamCode(null)
            onCreated?.()
          }}
          variant="outline"
        >
          Create Another Team
        </Button>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div>
        <label htmlFor="team-name" className="block text-sm font-medium text-gray-700 mb-2">
          Team Name
        </label>
        <input
          id="team-name"
          type="text"
          value={teamName}
          onChange={(e) => setTeamName(e.target.value)}
          placeholder="e.g., Varsity Tennis Team"
          className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm"
        />
      </div>
      {error && (
        <p className="text-sm text-red-600">{error}</p>
      )}
      <Button onClick={handleCreate} disabled={isCreating}>
        {isCreating ? 'Creating...' : 'Create Team'}
      </Button>
    </div>
  )
}
