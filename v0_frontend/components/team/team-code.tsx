"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useTeams } from "@/hooks/useTeams"

interface TeamCodeProps {
  onJoin?: () => void
}

export function TeamCode({ onJoin }: TeamCodeProps) {
  const { joinTeam, isJoining } = useTeams()
  const [code, setCode] = useState("")
  const [error, setError] = useState<string | null>(null)

  const handleJoin = async () => {
    if (!code.trim()) {
      setError("Please enter a team code")
      return
    }

    setError(null)

    try {
      await joinTeam(code.toUpperCase())
      setCode("")
      onJoin?.()
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to join team")
    }
  }

  return (
    <div className="space-y-4">
      <div>
        <Label htmlFor="code" className="text-gray-400 text-sm font-medium">
          Enter Team Code
        </Label>
        <div className="flex gap-2 mt-1">
          <Input
            id="code"
            type="text"
            value={code}
            onChange={(e) => setCode(e.target.value.toUpperCase())}
            placeholder="ABC123"
            maxLength={6}
            className="flex-1 bg-black/50 border-[#333333] text-white placeholder-gray-500 focus:border-[#50C878] focus:ring-[#50C878] uppercase"
          />
          <Button
            onClick={handleJoin}
            disabled={isJoining}
            className="bg-[#50C878] hover:bg-[#45b069] text-black font-semibold"
          >
            {isJoining ? "Joining..." : "Join"}
          </Button>
        </div>
      </div>

      {error && <p className="text-sm text-red-400 mt-2">{error}</p>}
    </div>
  )
}
