"use client"

import type React from "react"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useAuth } from "@/hooks/useAuth"
import { useProfile } from "@/hooks/useProfile"
import { createBrowserClient } from "@supabase/ssr"

interface UploadModalProps {
  isOpen: boolean
  onClose: () => void
}

const supabase = createBrowserClient(process.env.NEXT_PUBLIC_SUPABASE_URL!, process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!)

export function UploadModal({ isOpen, onClose }: UploadModalProps) {
  const router = useRouter()
  const { getUser } = useAuth()
  const { profile } = useProfile()

  const [playsightLink, setPlaysightLink] = useState("")
  const [playerName, setPlayerName] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

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

      const { data, error: insertError } = await supabase
        .from("matches")
        .insert([
          {
            user_id: user.id,
            playsight_link: playsightLink,
            player_name: playerName || null,
            status: "pending",
          },
        ])
        .select()
        .single()

      if (insertError) throw insertError

      onClose()
      router.push(`/matches/${data.id}/identify`)
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

          {profile?.role === "player" && (
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
