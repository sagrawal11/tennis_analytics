"use client"

import { MainLayout } from "@/components/layout/main-layout"
import { useMatches } from "@/hooks/useMatches"
import { useProfile } from "@/hooks/useProfile"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from "recharts"

const COLORS = ["#50C878", "#ef4444", "#3b82f6"]

export default function StatsPage() {
  const { data: matches, isLoading } = useMatches()
  const { profile } = useProfile()

  const completedMatches = matches?.filter((m) => m.status === "completed") || []
  const totalMatches = completedMatches.length
  const totalShots = 0 // Placeholder - would come from shots data
  const avgShotsPerMatch = totalMatches > 0 ? Math.round(totalShots / totalMatches) : 0

  // Placeholder chart data
  const shotDistribution = [
    { name: "Winners", value: 35 },
    { name: "Errors", value: 25 },
    { name: "In Play", value: 40 },
  ]

  const matchPerformance = completedMatches.slice(0, 5).map((match, i) => ({
    name: `Match ${i + 1}`,
    winners: Math.floor(Math.random() * 30) + 10,
    errors: Math.floor(Math.random() * 20) + 5,
  }))

  return (
    <MainLayout>
      <div className="mx-auto px-4 py-8 max-w-7xl">
        <h1 className="text-3xl font-bold text-white mb-6">Statistics</h1>

        {isLoading ? (
          <p className="text-gray-400">Loading statistics...</p>
        ) : totalMatches === 0 ? (
          <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-12 text-center shadow-xl">
            <p className="text-gray-400">No statistics yet. Upload matches to see your performance analytics.</p>
          </div>
        ) : (
          <>
            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
              <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-6 shadow-xl">
                <p className="text-sm font-medium text-gray-400 mb-2">Total Matches</p>
                <p className="text-3xl font-bold text-white">{totalMatches}</p>
              </div>
              <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-6 shadow-xl">
                <p className="text-sm font-medium text-gray-400 mb-2">Total Shots</p>
                <p className="text-3xl font-bold text-white">{totalShots}</p>
              </div>
              <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-6 shadow-xl">
                <p className="text-sm font-medium text-gray-400 mb-2">Win Rate</p>
                <p className="text-3xl font-bold text-[#50C878]">58%</p>
              </div>
              <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-6 shadow-xl">
                <p className="text-sm font-medium text-gray-400 mb-2">Avg Shots/Match</p>
                <p className="text-3xl font-bold text-white">{avgShotsPerMatch}</p>
              </div>
            </div>

            {/* Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Shot Distribution */}
              <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-6 shadow-xl">
                <h3 className="text-lg font-semibold text-white mb-4">Shot Distribution</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={shotDistribution}
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      dataKey="value"
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    >
                      {shotDistribution.map((_, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#1a1a1a",
                        border: "1px solid #333",
                        borderRadius: "8px",
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              {/* Performance by Match */}
              <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-6 shadow-xl">
                <h3 className="text-lg font-semibold text-white mb-4">Performance by Match</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={matchPerformance}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                    <XAxis dataKey="name" stroke="#a0a0a0" />
                    <YAxis stroke="#a0a0a0" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#1a1a1a",
                        border: "1px solid #333",
                        borderRadius: "8px",
                      }}
                    />
                    <Bar dataKey="winners" fill="#50C878" name="Winners" />
                    <Bar dataKey="errors" fill="#ef4444" name="Errors" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </>
        )}
      </div>
    </MainLayout>
  )
}
