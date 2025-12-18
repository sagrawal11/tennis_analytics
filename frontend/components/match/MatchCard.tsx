import Link from 'next/link'

interface MatchCardProps {
  match: {
    id: string
    player_name?: string
    status: string
    created_at: string
    playsight_link: string
  }
}

export function MatchCard({ match }: MatchCardProps) {
  const statusColors = {
    pending: 'bg-yellow-100 text-yellow-800',
    processing: 'bg-blue-100 text-blue-800',
    completed: 'bg-green-100 text-green-800',
    failed: 'bg-red-100 text-red-800',
  }

  const statusColor = statusColors[match.status as keyof typeof statusColors] || 'bg-gray-100 text-gray-800'

  return (
    <Link href={`/matches/${match.id}`}>
      <div className="bg-white rounded-lg shadow p-6 hover:shadow-lg transition-shadow cursor-pointer">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">
            {match.player_name || 'Match'}
          </h3>
          <span className={`px-2 py-1 rounded text-xs font-medium ${statusColor}`}>
            {match.status}
          </span>
        </div>
        <p className="text-sm text-gray-600 mb-2">
          {new Date(match.created_at).toLocaleDateString()}
        </p>
        <p className="text-xs text-gray-500 truncate">
          {match.playsight_link}
        </p>
      </div>
    </Link>
  )
}
