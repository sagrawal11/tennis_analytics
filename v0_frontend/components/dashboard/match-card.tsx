import Link from "next/link"
import type { Match } from "@/lib/types"

interface MatchCardProps {
  match: Match
}

const statusColors = {
  pending: "bg-amber-900/30 text-amber-400 border-amber-800",
  processing: "bg-blue-900/30 text-blue-400 border-blue-800",
  completed: "bg-emerald-900/30 text-emerald-400 border-emerald-800",
  failed: "bg-red-900/30 text-red-400 border-red-800",
}

export function MatchCard({ match }: MatchCardProps) {
  return (
    <Link href={`/matches/${match.id}`}>
      <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-6 hover:border-[#50C878]/50 hover:shadow-lg hover:shadow-[#50C878]/10 transition-all cursor-pointer">
        <div className="flex justify-between items-start mb-2">
          <h3 className="text-lg font-semibold text-white">{match.player_name || "Match"}</h3>
          <span className={`px-2 py-1 rounded text-xs font-medium border ${statusColors[match.status]}`}>
            {match.status}
          </span>
        </div>
        <p className="text-sm text-gray-400 mb-2">{new Date(match.created_at).toLocaleDateString()}</p>
        <p className="text-xs text-gray-500 truncate">{match.playsight_link}</p>
      </div>
    </Link>
  )
}
