"use client"

import { useState } from "react"
import { MainLayout } from "@/components/layout/main-layout"
import { MatchCard } from "@/components/dashboard/match-card"
import { useMatches } from "@/hooks/useMatches"
import type { Match } from "@/lib/types"

export default function DashboardPage() {
  const { data: matches, isLoading, error } = useMatches()
  const [expandedDate, setExpandedDate] = useState<string | null>(null)

  // Group matches by date
  const groupedMatches = matches?.reduce(
    (groups, match) => {
      const date = new Date(match.created_at).toLocaleDateString()
      if (!groups[date]) {
        groups[date] = []
      }
      groups[date].push(match)
      return groups
    },
    {} as Record<string, Match[]>,
  )

  const sortedDates = groupedMatches
    ? Object.keys(groupedMatches).sort((a, b) => new Date(b).getTime() - new Date(a).getTime())
    : []

  return (
    <MainLayout>
      <div className="mx-auto px-4 py-8 max-w-7xl">
        <h1 className="text-3xl font-bold text-white mb-6">Dashboard</h1>

        {isLoading && <div className="text-center text-gray-400 py-12">Loading matches...</div>}

        {error && (
          <div className="bg-red-900/20 border-2 border-red-800 rounded-lg p-4">
            <p className="text-red-400">Error loading matches. Please try again.</p>
          </div>
        )}

        {!isLoading && !error && (!matches || matches.length === 0) && (
          <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-12 text-center shadow-xl">
            <h2 className="text-xl font-semibold text-white mb-2">Welcome to Courtvision!</h2>
            <p className="text-gray-400 mb-4">Upload your first match to see detailed analytics and insights.</p>
            <p className="text-sm text-gray-500">Click the + button in the bottom right to get started.</p>
          </div>
        )}

        {!isLoading && !error && sortedDates.length > 0 && (
          <div className="space-y-4">
            {sortedDates.map((date) => {
              const dateMatches = groupedMatches![date]
              const isExpanded = expandedDate === date

              return (
                <div key={date} className="bg-[#1a1a1a] rounded-lg border border-[#333333] shadow-xl overflow-hidden">
                  <button
                    onClick={() => setExpandedDate(isExpanded ? null : date)}
                    className="w-full flex justify-between items-center px-6 py-4 hover:bg-[#262626]/50 transition-colors"
                  >
                    <span className="text-lg font-semibold text-white">{date}</span>
                    <span className="text-sm text-gray-400">
                      {dateMatches.length} {dateMatches.length === 1 ? "match" : "matches"}
                    </span>
                  </button>

                  {isExpanded && (
                    <div className="px-6 pb-4">
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {dateMatches.map((match) => (
                          <MatchCard key={match.id} match={match} />
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </div>
    </MainLayout>
  )
}
