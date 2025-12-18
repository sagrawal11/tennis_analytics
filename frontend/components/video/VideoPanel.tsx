'use client'

import { X } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface VideoPanelProps {
  videoUrl: string
  timestamp: number
  onClose: () => void
}

export function VideoPanel({ videoUrl, timestamp, onClose }: VideoPanelProps) {
  // For Playsight videos, we'll embed them
  // The timestamp will be handled by Playsight's player if it supports it
  const embedUrl = videoUrl.includes('playsight') 
    ? videoUrl 
    : videoUrl

  return (
    <div className="bg-white rounded-lg shadow p-4 sticky top-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">Video</h3>
        <Button
          variant="ghost"
          size="icon"
          onClick={onClose}
          className="h-8 w-8"
        >
          <X className="h-4 w-4" />
        </Button>
      </div>
      <div className="aspect-video bg-black rounded overflow-hidden">
        {videoUrl.includes('playsight') ? (
          <iframe
            src={embedUrl}
            className="w-full h-full"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowFullScreen
          />
        ) : (
          <video
            src={videoUrl}
            controls
            className="w-full h-full"
            onLoadedMetadata={(e) => {
              const video = e.currentTarget
              if (timestamp > 0) {
                video.currentTime = timestamp
              }
            }}
          />
        )}
      </div>
      {timestamp > 0 && (
        <p className="text-sm text-gray-600 mt-2">
          Jump to: {Math.floor(timestamp / 60)}:{(timestamp % 60).toFixed(0).padStart(2, '0')}
        </p>
      )}
    </div>
  )
}
