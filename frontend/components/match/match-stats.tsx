import type { Match, Shot } from "@/lib/types"

interface MatchStatsProps {
  match: Match
  matchData?: unknown
  shots?: Shot[]
}

export function MatchStats({ shots = [] }: MatchStatsProps) {
  const winners = shots.filter((s) => s.result === "winner").length
  const errors = shots.filter((s) => s.result === "error").length
  const inPlay = shots.filter((s) => s.result === "in_play").length
  const total = shots.length

  const stats = [
    { label: "Total Shots", value: total },
    { label: "Winners", value: winners, color: "text-green-400" },
    { label: "Errors", value: errors, color: "text-red-400" },
    { label: "In Play", value: inPlay, color: "text-blue-400" },
    {
      label: "Winner %",
      value: total > 0 ? `${Math.round((winners / total) * 100)}%` : "0%",
      color: "text-[#50C878]",
    },
  ]

  return (
    <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-6 shadow-xl">
      <h3 className="text-lg font-semibold text-white mb-4">Match Statistics</h3>
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        {stats.map((stat) => (
          <div key={stat.label} className="text-center">
            <p className={`text-2xl font-bold ${stat.color || "text-white"}`}>{stat.value}</p>
            <p className="text-sm text-gray-400">{stat.label}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
