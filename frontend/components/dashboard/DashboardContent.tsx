'use client'

import { useMatches } from '@/hooks/useMatches'
import { MatchCard } from '@/components/match/MatchCard'
import { useState } from 'react'

interface DashboardContentProps {
  user: {
    id: string
    email?: string
  }
  profile: {
    name?: string
    role?: string
  } | null
}

export function DashboardContent({ user, profile }: DashboardContentProps) {
  const { data: matches, isLoading, error } = useMatches()
  const [expandedDate, setExpandedDate] = useState<string | null>(null)

  // Group matches by date
  const matchesByDate = matches?.reduce((acc, match) => {
    const date = new Date(match.created_at).toLocaleDateString()
    if (!acc[date]) {
      acc[date] = []
    }
    acc[date].push(match)
    return acc
  }, {} as Record<string, typeof matches>) || {}

  const isCoach = profile?.role === 'coach'

  if (isLoading) {
    return (
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold mb-6">Dashboard</h1>
        <div className="text-center py-12">
          <p className="text-gray-600">Loading matches...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold mb-6">Dashboard</h1>
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-800">Error loading matches. Please try again.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-6">Dashboard</h1>

      {matches && matches.length === 0 ? (
        <div className="bg-white rounded-lg shadow p-12 text-center">
          <p className="text-gray-600 mb-4">No matches yet.</p>
          <p className="text-sm text-gray-500">
            Click the + button to upload your first match video.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {Object.entries(matchesByDate).map(([date, dateMatches]) => (
            <div key={date} className="bg-white rounded-lg shadow">
              <button
                onClick={() =>
                  setExpandedDate(expandedDate === date ? null : date)
                }
                className="w-full px-6 py-4 flex items-center justify-between hover:bg-gray-50 transition-colors"
              >
                <h2 className="text-lg font-semibold">{date}</h2>
                <span className="text-sm text-gray-500">
                  {dateMatches.length} match{dateMatches.length !== 1 ? 'es' : ''}
                </span>
              </button>
              {expandedDate === date && (
                <div className="px-6 pb-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {dateMatches.map((match) => (
                    <MatchCard key={match.id} match={match} />
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
