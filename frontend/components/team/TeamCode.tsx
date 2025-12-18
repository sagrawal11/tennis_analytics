'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { useTeams } from '@/hooks/useTeams'

interface TeamCodeProps {
  onJoin?: () => void
}

export function TeamCode({ onJoin }: TeamCodeProps) {
  const [code, setCode] = useState('')
  const [error, setError] = useState<string | null>(null)
  const { joinTeam, isJoining } = useTeams()

  const handleJoin = async () => {
    if (!code.trim()) {
      setError('Please enter a team code')
      return
    }

    setError(null)

    try {
      await joinTeam(code.toUpperCase())
      setCode('')
      onJoin?.()
    } catch (err: any) {
      setError(err.message || 'Failed to join team')
    }
  }

  return (
    <div className="space-y-4">
      <div>
        <label htmlFor="team-code" className="block text-sm font-medium text-gray-700 mb-2">
          Enter Team Code
        </label>
        <div className="flex gap-2">
          <input
            id="team-code"
            type="text"
            value={code}
            onChange={(e) => setCode(e.target.value.toUpperCase())}
            placeholder="ABC123"
            maxLength={6}
            className="flex-1 rounded-md border border-gray-300 px-3 py-2 text-sm uppercase"
          />
          <Button onClick={handleJoin} disabled={isJoining}>
            {isJoining ? 'Joining...' : 'Join'}
          </Button>
        </div>
        {error && (
          <p className="mt-2 text-sm text-red-600">{error}</p>
        )}
      </div>
    </div>
  )
}
