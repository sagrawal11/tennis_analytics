"use client"

import { useEffect, useState } from "react"
import { useParams } from "next/navigation"
import { MainLayout } from "@/components/layout/main-layout"
import { CourtDiagram } from "@/components/court/court-diagram"
import { VideoPanel } from "@/components/video/video-panel"
import { ProcessingStatus } from "@/components/upload/processing-status"
import { MatchStats } from "@/components/match/match-stats"
import { createBrowserClient } from "@supabase/ssr"
import type { Match, Shot } from "@/lib/types"

const supabase = createBrowserClient(process.env.NEXT_PUBLIC_SUPABASE_URL!, process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!)

type CourtShot = {
  id: string
  start_pos: { x: number; y: number }
  end_pos: { x: number; y: number }
  result: "winner" | "error" | "in_play"
  video_timestamp: number | null
}

export default function MatchDetailPage() {
  const params = useParams()
  const matchId = params.id as string

  const [match, setMatch] = useState<Match | null>(null)
  const [shots, setShots] = useState<Shot[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedShot, setSelectedShot] = useState<CourtShot | null>(null)
  const [showVideoPanel, setShowVideoPanel] = useState(false)

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true)
      try {
        const { data: matchData } = await supabase.from("matches").select("*").eq("id", matchId).single()

        setMatch(matchData)

        if (matchData?.status === "completed") {
          const { data: shotsData } = await supabase.from("shots").select("*").eq("match_id", matchId)

          setShots(shotsData || [])
        }
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [matchId])

  const handleShotClick = (shot: CourtShot) => {
    setSelectedShot(shot)
    setShowVideoPanel(true)
  }

  const courtShots: CourtShot[] = shots.map((shot) => ({
    id: shot.id,
    start_pos: shot.start_pos,
    end_pos: shot.end_pos,
    result: shot.result,
    video_timestamp: shot.video_timestamp,
  }))

  if (loading) {
    return (
      <MainLayout>
        <div className="mx-auto px-4 py-8 max-w-7xl">
          <p className="text-gray-400">Loading match...</p>
        </div>
      </MainLayout>
    )
  }

  if (!match) {
    return (
      <MainLayout>
        <div className="mx-auto px-4 py-8 max-w-7xl">
          <p className="text-red-400">Match not found</p>
        </div>
      </MainLayout>
    )
  }

  return (
    <MainLayout>
      <div className="mx-auto px-4 py-8 max-w-7xl">
        <h1 className="text-3xl font-bold text-white mb-2">
          {match.player_name || "Match"} - {new Date(match.created_at).toLocaleDateString()}
        </h1>

        {match.status !== "completed" && (
          <div className="mt-4">
            <ProcessingStatus matchId={match.id} />
          </div>
        )}

        <div className="mt-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Court Diagram - 2 columns */}
          <div className="lg:col-span-2">
            {match.status === "completed" ? (
              <>
                <h2 className="text-xl font-semibold text-white mb-4">Court Visualization</h2>
                <CourtDiagram shots={courtShots} onShotClick={handleShotClick} />
              </>
            ) : (
              <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-12 text-center">
                <p className="text-gray-400">
                  Processing in progress. Court visualization will appear when processing is complete.
                </p>
              </div>
            )}
          </div>

          {/* Video Panel - 1 column */}
          {showVideoPanel && selectedShot && (
            <div className="lg:col-span-1">
              <VideoPanel
                videoUrl={match.video_url || match.playsight_link}
                timestamp={selectedShot.video_timestamp || 0}
                onClose={() => setShowVideoPanel(false)}
              />
            </div>
          )}
        </div>

        {/* Match Stats */}
        <div className="mt-6">
          <MatchStats match={match} shots={shots} />
        </div>
      </div>
    </MainLayout>
  )
}
