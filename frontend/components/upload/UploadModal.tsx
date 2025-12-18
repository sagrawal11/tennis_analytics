'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { X } from 'lucide-react'
import { useAuth } from '@/hooks/useAuth'
import { createClient } from '@/lib/supabase/client'
import { useRouter } from 'next/navigation'

interface UploadModalProps {
  isOpen: boolean
  onClose: () => void
}

export function UploadModal({ isOpen, onClose }: UploadModalProps) {
  const [playsightLink, setPlaysightLink] = useState('')
  const [playerName, setPlayerName] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const { getUser } = useAuth()
  const supabase = createClient()
  const router = useRouter()

  if (!isOpen) return null

  const validatePlaysightLink = (link: string): boolean => {
    // Basic validation - can be enhanced
    return link.includes('playsight') || link.startsWith('http')
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)

    if (!playsightLink.trim()) {
      setError('Please enter a Playsight link')
      return
    }

    if (!validatePlaysightLink(playsightLink)) {
      setError('Please enter a valid Playsight link')
      return
    }

    setLoading(true)

    try {
      const user = await getUser()
      if (!user) {
        setError('Please sign in first')
        return
      }

      const { data: { session } } = await supabase.auth.getSession()
      if (!session) {
        setError('Please sign in first')
        return
      }

      // Create match
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/matches`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${session.access_token}`,
        },
        body: JSON.stringify({
          playsight_link: playsightLink,
          player_name: playerName || undefined,
        }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to create match')
      }

      // Close modal and redirect to player identification
      onClose()
      router.push(`/matches/${data.match.id}/identify`)
    } catch (err: any) {
      setError(err.message || 'Failed to upload video')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">Upload Match Video</h2>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="playsight-link" className="block text-sm font-medium text-gray-700 mb-2">
              Playsight Link
            </label>
            <input
              id="playsight-link"
              type="url"
              value={playsightLink}
              onChange={(e) => setPlaysightLink(e.target.value)}
              placeholder="https://playsight.com/..."
              className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm"
              required
            />
          </div>

          <div>
            <label htmlFor="player-name" className="block text-sm font-medium text-gray-700 mb-2">
              Player Name (Optional)
            </label>
            <input
              id="player-name"
              type="text"
              value={playerName}
              onChange={(e) => setPlayerName(e.target.value)}
              placeholder="Player name"
              className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm"
            />
          </div>

          {error && (
            <div className="bg-red-50 border border-red-200 rounded p-3">
              <p className="text-sm text-red-800">{error}</p>
            </div>
          )}

          <div className="flex gap-2 justify-end">
            <Button type="button" variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" disabled={loading}>
              {loading ? 'Uploading...' : 'Upload'}
            </Button>
          </div>
        </form>
      </div>
    </div>
  )
}
