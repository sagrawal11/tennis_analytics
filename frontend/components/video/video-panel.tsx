"use client"

import { X } from "lucide-react"
import { Button } from "@/components/ui/button"

interface VideoPanelProps {
  videoUrl: string
  timestamp: number
  onClose: () => void
}

export function VideoPanel({ videoUrl, timestamp, onClose }: VideoPanelProps) {
  // Convert Playsight link to embed URL if needed
  const embedUrl = videoUrl.includes("playsight") ? videoUrl.replace("/video/", "/embed/") : videoUrl

  const formatTimestamp = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, "0")}`
  }

  return (
    <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-4 shadow-xl sticky top-4">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold text-white">Video</h3>
        <Button
          variant="ghost"
          size="icon"
          onClick={onClose}
          className="text-gray-400 hover:text-white hover:bg-[#262626]"
        >
          <X className="h-5 w-5" />
        </Button>
      </div>

      <div className="aspect-video bg-black rounded overflow-hidden">
        {videoUrl.includes("playsight") ? (
          <iframe src={embedUrl} className="w-full h-full" allowFullScreen title="Match Video" />
        ) : (
          <video
            src={videoUrl}
            controls
            className="w-full h-full"
            onLoadedMetadata={(e) => {
              const video = e.currentTarget
              video.currentTime = timestamp
            }}
          />
        )}
      </div>

      {timestamp > 0 && <p className="text-sm text-gray-500 mt-2">Jump to: {formatTimestamp(timestamp)}</p>}
    </div>
  )
}
