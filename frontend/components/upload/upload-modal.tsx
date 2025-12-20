"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { useAuth } from "@/hooks/useAuth"
import { useProfile } from "@/hooks/useProfile"
import { useTeams } from "@/hooks/useTeams"
import { useTeamMembers } from "@/hooks/useTeamMembers"
import { createClient } from "@/lib/supabase/client"

interface UploadModalProps {
  isOpen: boolean
  onClose: () => void
}

export function UploadModal({ isOpen, onClose }: UploadModalProps) {
  const router = useRouter()
  const { getUser } = useAuth()
  const { profile } = useProfile()
  const { teams } = useTeams()
  const supabase = createClient()

  const [playsightLink, setPlaysightLink] = useState("")
  const [playerName, setPlayerName] = useState("")
  const [selectedPlayerId, setSelectedPlayerId] = useState<string>("")
  const [teamMembers, setTeamMembers] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const isCoach = profile?.role === "coach"

  // Fetch team members when coach selects a team
  useEffect(() => {
    const fetchTeamMembers = async () => {
      if (!isCoach || teams.length === 0) {
        setTeamMembers([])
        return
      }

      // Get all team members from all teams
      const allMembers: any[] = []
      for (const team of teams) {
        const { data: { session } } = await supabase.auth.getSession()
        if (!session) continue

        try {
          const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/teams/${team.id}/members`, {
            headers: {
              'Authorization': `Bearer ${session.access_token}`,
            },
          })

          if (response.ok) {
            const data = await response.json()
            const members = (data.members || []).filter((m: any) => m.users?.role === 'player')
            allMembers.push(...members.map((m: any) => ({
              id: m.users?.id,
              name: m.users?.name || m.users?.email || 'Unknown',
              email: m.users?.email,
            })))
          }
        } catch (err) {
          console.error('Error fetching team members:', err)
        }
      }

      // Remove duplicates
      const uniqueMembers = Array.from(
        new Map(allMembers.map(m => [m.id, m])).values()
      )
      setTeamMembers(uniqueMembers)
    }

    fetchTeamMembers()
  }, [isCoach, teams, supabase])

  if (!isOpen) return null

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)

    if (!playsightLink.trim()) {
      setError("Please enter a Playsight link")
      return
    }

    setLoading(true)

    try {
      const user = await getUser()
      if (!user) {
        setError("Please sign in first")
        return
      }

      const { data: { session } } = await supabase.auth.getSession()
      if (!session) {
        setError("Please sign in first")
        return
      }

      // Determine user_id: if coach selected a player, use that; otherwise use current user
      const matchUserId = (isCoach && selectedPlayerId) ? selectedPlayerId : user.id

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/matches`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${session.access_token}`,
        },
        body: JSON.stringify({
          playsight_link: playsightLink,
          player_name: playerName || undefined,
          user_id: (isCoach && selectedPlayerId) ? selectedPlayerId : undefined,
        }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to create match')
      }

      onClose()
      router.push(`/matches/${data.match.id}/identify`)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to upload video")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-[#1a1a1a] max-w-md w-full rounded-2xl p-6 border border-[#333333] shadow-2xl">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold text-white">Upload Match Video</h2>
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            className="text-gray-400 hover:text-white hover:bg-[#262626]"
          >
            <X className="h-5 w-5" />
          </Button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <Label htmlFor="playsightLink" className="text-gray-400 text-sm font-medium">
              Playsight Link
            </Label>
            <Input
              id="playsightLink"
              type="url"
              value={playsightLink}
              onChange={(e) => setPlaysightLink(e.target.value)}
              placeholder="https://playsight.com/..."
              className="mt-1 bg-black/50 border-[#333333] text-white placeholder-gray-500 focus:border-[#50C878] focus:ring-[#50C878]"
              required
            />
          </div>

          {isCoach && teamMembers.length > 0 && (
            <div>
              <Label htmlFor="playerSelect" className="text-gray-400 text-sm font-medium">
                Select Player
              </Label>
              <Select value={selectedPlayerId} onValueChange={setSelectedPlayerId}>
                <SelectTrigger className="mt-1 bg-black/50 border-[#333333] text-white focus:border-[#50C878] focus:ring-[#50C878]">
                  <SelectValue placeholder="Select a player" />
                </SelectTrigger>
                <SelectContent className="bg-[#1a1a1a] border-[#333333]">
                  {teamMembers.map((member) => (
                    <SelectItem key={member.id} value={member.id} className="text-white hover:bg-[#262626]">
                      {member.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}

          {!isCoach && (
            <div>
              <Label htmlFor="playerName" className="text-gray-400 text-sm font-medium">
                Player Name (Optional)
              </Label>
              <Input
                id="playerName"
                type="text"
                value={playerName}
                onChange={(e) => setPlayerName(e.target.value)}
                placeholder="Player name"
                className="mt-1 bg-black/50 border-[#333333] text-white placeholder-gray-500 focus:border-[#50C878] focus:ring-[#50C878]"
              />
            </div>
          )}

          {error && (
            <div className="bg-red-900/20 border border-red-800 rounded-lg p-3">
              <p className="text-sm text-red-300">{error}</p>
            </div>
          )}

          <div className="flex gap-2 justify-end">
            <Button
              type="button"
              variant="outline"
              onClick={onClose}
              className="border-[#333333] text-gray-300 hover:border-[#50C878] hover:text-white bg-transparent"
            >
              Cancel
            </Button>
            <Button
              type="submit"
              disabled={loading}
              className="bg-[#50C878] hover:bg-[#45b069] text-black font-semibold"
            >
              {loading ? "Uploading..." : "Upload"}
            </Button>
          </div>
        </form>
      </div>
    </div>
  )
}
