"use client"

import { useState } from "react"
import { useQueryClient } from "@tanstack/react-query"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useTeams } from "@/hooks/useTeams"
import { useProfile } from "@/hooks/useProfile"

interface TeamCodeProps {
  onJoin?: () => void
}

export function TeamCode({ onJoin }: TeamCodeProps) {
  const { joinTeam, isJoining } = useTeams()
  const { profile } = useProfile()
  const queryClient = useQueryClient()
  const [code, setCode] = useState("")
  const [error, setError] = useState<string | null>(null)
  const [autoActivated, setAutoActivated] = useState(false)

  const isCoach = profile?.role === "coach"

  const handleJoin = async () => {
    if (!code.trim()) {
      setError("Please enter a team code")
      return
    }

    setError(null)
    setAutoActivated(false)

    try {
      const result = await joinTeam(code.trim().toUpperCase())
      setCode("")
      
      // Refresh all related queries after joining (with a delay to ensure backend has processed)
      // Do this regardless of auto_activated flag, in case the backend updated but didn't return the flag
      setTimeout(() => {
        queryClient.invalidateQueries({ queryKey: ['my-teams'] })
        queryClient.invalidateQueries({ queryKey: ['team-activation-status'] })
        queryClient.invalidateQueries({ queryKey: ['activation-status'] })
        queryClient.invalidateQueries({ queryKey: ['profile'] })
      }, 500) // Increased delay to ensure backend update is complete
      
      // Also do an immediate refresh for activation status
      queryClient.invalidateQueries({ queryKey: ['activation-status'] })
      queryClient.invalidateQueries({ queryKey: ['profile'] })
      
      // If coach or player was auto-activated, show message
      if (result?.auto_activated) {
        setAutoActivated(true)
        setTimeout(() => setAutoActivated(false), 5000) // Hide after 5 seconds
      }
      
      onJoin?.()
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : "Failed to join team"
      setError(errorMessage)
      console.error('Join team error:', err)
      
      // If it's a network error, suggest checking connection
      if (errorMessage.includes('Network error') || errorMessage.includes('fetch')) {
        // Still invalidate queries in case the request actually succeeded
        setTimeout(() => {
          queryClient.invalidateQueries({ queryKey: ['my-teams'] })
          queryClient.invalidateQueries({ queryKey: ['team-activation-status'] })
        }, 1000)
      }
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
      {autoActivated && (
        <div className="bg-emerald-900/20 border-2 border-emerald-800 rounded-lg p-3 mt-2">
          <p className="text-sm text-emerald-400 font-medium">
            ✓ Account activated! The team has an activated coach, so your account has been automatically activated.
          </p>
        </div>
      )}
    </div>
  )
}
