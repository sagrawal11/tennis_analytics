'use client'

import { CourtDiagram } from '@/components/court/CourtDiagram'
import { VideoPanel } from '@/components/video/VideoPanel'
import { MatchStats } from '@/components/stats/MatchStats'
import { ProcessingStatus } from '@/components/upload/ProcessingStatus'
import { useState } from 'react'

interface MatchDetailContentProps {
  match: any
  matchData: any
  shots: any[]
}

export function MatchDetailContent({
  match,
  matchData,
  shots,
}: MatchDetailContentProps) {
  const [selectedShot, setSelectedShot] = useState<any>(null)
  const [showVideoPanel, setShowVideoPanel] = useState(false)

  const handleShotClick = (shot: any) => {
    setSelectedShot(shot)
    setShowVideoPanel(true)
  }

  // Transform shots for court diagram
  const courtShots = shots.map((shot) => ({
    id: shot.id,
    start_pos: shot.start_pos,
    end_pos: shot.end_pos,
    result: shot.result,
    video_timestamp: shot.video_timestamp,
  }))

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">
          {match.player_name || 'Match'} - {new Date(match.created_at).toLocaleDateString()}
        </h1>
        {match.status !== 'completed' && (
          <div className="mt-4">
            <ProcessingStatus matchId={match.id} />
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Court Diagram - takes 2 columns */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold mb-4">Court Visualization</h2>
            <CourtDiagram shots={courtShots} onShotClick={handleShotClick} />
            <div className="mt-4 flex gap-4 text-sm">
              <div className="flex items-center gap-2">
                <div className="w-4 h-0.5 bg-green-500"></div>
                <span>Winners</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-0.5 bg-red-500 border-dashed border-t-2 border-red-500"></div>
                <span>Errors</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-0.5 bg-blue-500"></div>
                <span>In Play</span>
              </div>
            </div>
          </div>
        </div>

        {/* Video Panel - takes 1 column */}
        <div className="lg:col-span-1">
          {showVideoPanel && selectedShot && (
            <VideoPanel
              videoUrl={match.video_url || match.playsight_link}
              timestamp={selectedShot.video_timestamp || 0}
              onClose={() => setShowVideoPanel(false)}
            />
          )}
        </div>
      </div>

      {/* Stats Section */}
      <div className="mt-6">
        <MatchStats match={match} matchData={matchData} shots={shots} />
      </div>
    </div>
  )
}
