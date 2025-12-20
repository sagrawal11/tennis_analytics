"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useTeams } from "@/hooks/useTeams"

interface CreateTeamProps {
  onCreated?: () => void
}

export function CreateTeam({ onCreated }: CreateTeamProps) {
  const { createTeam, isCreating } = useTeams()
  const [teamName, setTeamName] = useState("")
  const [teamCode, setTeamCode] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleCreate = async () => {
    if (!teamName.trim()) {
      setError("Please enter a team name")
      return
    }

    setError(null)

    try {
      const data = await createTeam(teamName)
      setTeamCode(data.code)
      setTeamName("")
      onCreated?.()
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to create team")
    }
  }

  if (teamCode) {
    return (
      <div className="bg-emerald-900/20 border-2 border-emerald-800 rounded-lg p-4">
        <p className="text-sm font-medium text-emerald-400 mb-2">Team created successfully!</p>
        <p className="text-sm text-gray-400 mb-4">Share this code with your players:</p>
        <div className="bg-black/50 rounded border-2 border-[#50C878] p-4 text-center">
          <span className="text-3xl font-bold text-[#50C878] font-mono">{teamCode}</span>
        </div>
        <Button
          variant="outline"
          onClick={() => setTeamCode(null)}
          className="mt-4 border-[#333333] text-white hover:border-[#50C878]"
        >
          Create Another Team
        </Button>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div>
        <Label htmlFor="teamName" className="text-gray-400 text-sm font-medium">
          Team Name
        </Label>
        <Input
          id="teamName"
          type="text"
          value={teamName}
          onChange={(e) => setTeamName(e.target.value)}
          placeholder="e.g., Varsity Tennis Team"
          className="mt-1 bg-black/50 border-[#333333] text-white placeholder-gray-500 focus:border-[#50C878] focus:ring-[#50C878]"
        />
      </div>

      {error && <p className="text-sm text-red-400">{error}</p>}

      <Button
        onClick={handleCreate}
        disabled={isCreating}
        className="bg-[#50C878] hover:bg-[#45b069] text-black font-semibold"
      >
        {isCreating ? "Creating..." : "Create Team"}
      </Button>
    </div>
  )
}
