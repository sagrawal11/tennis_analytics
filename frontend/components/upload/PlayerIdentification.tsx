'use client'

import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { useAuth } from '@/hooks/useAuth'
import { createClient } from '@/lib/supabase/client'
import { useRouter } from 'next/navigation'

interface PlayerIdentificationProps {
  matchId: string
  playsightLink: string
}

export function PlayerIdentification({ matchId, playsightLink }: PlayerIdentificationProps) {
  const [frames, setFrames] = useState<string[]>([])
  const [selectedCoords, setSelectedCoords] = useState<Array<{ x: number; y: number }>>([])
  const [loading, setLoading] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const { getUser } = useAuth()
  const supabase = createClient()
  const router = useRouter()

  // Extract frames from video (placeholder - will need backend integration)
  useEffect(() => {
    // TODO: Call backend to extract frames from Playsight video
    // For now, show placeholder
    setFrames([
      '/placeholder-frame-1.jpg',
      '/placeholder-frame-2.jpg',
      '/placeholder-frame-3.jpg',
    ])
  }, [playsightLink])

  const handleFrameClick = (frameIndex: number, event: React.MouseEvent<HTMLImageElement>) => {
    const rect = event.currentTarget.getBoundingClientRect()
    const x = ((event.clientX - rect.left) / rect.width) * 100
    const y = ((event.clientY - rect.top) / rect.height) * 100

    const newCoords = [...selectedCoords]
    newCoords[frameIndex] = { x, y }
    setSelectedCoords(newCoords)
  }

  const handleSubmit = async () => {
    if (selectedCoords.length === 0) {
      setError('Please click on yourself in at least one frame')
      return
    }

    setSubmitting(true)
    setError(null)

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

      // Submit identification
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/videos/identify-player`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${session.access_token}`,
        },
        body: JSON.stringify({
          match_id: matchId,
          frame_data: {
            frames: frames,
            // In real implementation, this would be frame image data
          },
          selected_player_coords: selectedCoords,
        }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to submit identification')
      }

      // Redirect to match detail page
      router.push(`/matches/${matchId}`)
    } catch (err: any) {
      setError(err.message || 'Failed to submit identification')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <p className="text-gray-600 mb-6">
        Click on yourself in each frame to help us track your performance throughout the match.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {frames.map((frame, index) => (
          <div key={index} className="relative">
            <div className="aspect-video bg-gray-200 rounded border-2 border-dashed border-gray-300 flex items-center justify-center">
              {frame.startsWith('/placeholder') ? (
                <div className="text-center p-4">
                  <p className="text-sm text-gray-500">Frame {index + 1}</p>
                  <p className="text-xs text-gray-400 mt-2">Frame extraction coming soon</p>
                </div>
              ) : (
                <img
                  src={frame}
                  alt={`Frame ${index + 1}`}
                  className="w-full h-full object-cover rounded cursor-crosshair"
                  onClick={(e) => handleFrameClick(index, e)}
                />
              )}
            </div>
            {selectedCoords[index] && (
              <div
                className="absolute w-4 h-4 bg-blue-500 rounded-full border-2 border-white shadow-lg"
                style={{
                  left: `${selectedCoords[index].x}%`,
                  top: `${selectedCoords[index].y}%`,
                  transform: 'translate(-50%, -50%)',
                }}
              />
            )}
            <p className="text-xs text-gray-500 mt-2 text-center">
              {selectedCoords[index] ? '✓ Selected' : 'Click to select'}
            </p>
          </div>
        ))}
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded p-3 mb-4">
          <p className="text-sm text-red-800">{error}</p>
        </div>
      )}

      <div className="flex justify-end">
        <Button onClick={handleSubmit} disabled={submitting || selectedCoords.length === 0}>
          {submitting ? 'Submitting...' : 'Submit & Start Processing'}
        </Button>
      </div>
    </div>
  )
}
