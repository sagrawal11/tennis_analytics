'use client'

interface MatchStatsProps {
  match: any
  matchData: any
  shots: any[]
}

export function MatchStats({ match, matchData, shots }: MatchStatsProps) {
  // Calculate stats from shots
  const winners = shots.filter((s) => s.result === 'winner').length
  const errors = shots.filter((s) => s.result === 'error').length
  const inPlay = shots.filter((s) => s.result === 'in_play').length
  const total = shots.length

  // Get stats from match_data if available
  const statsSummary = matchData?.stats_summary || {}

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-xl font-semibold mb-4">Match Statistics</h2>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-green-50 rounded-lg p-4">
          <p className="text-sm text-gray-600 mb-1">Winners</p>
          <p className="text-2xl font-bold text-green-700">{winners}</p>
          {total > 0 && (
            <p className="text-xs text-gray-500">
              {((winners / total) * 100).toFixed(1)}%
            </p>
          )}
        </div>
        <div className="bg-red-50 rounded-lg p-4">
          <p className="text-sm text-gray-600 mb-1">Errors</p>
          <p className="text-2xl font-bold text-red-700">{errors}</p>
          {total > 0 && (
            <p className="text-xs text-gray-500">
              {((errors / total) * 100).toFixed(1)}%
            </p>
          )}
        </div>
        <div className="bg-blue-50 rounded-lg p-4">
          <p className="text-sm text-gray-600 mb-1">In Play</p>
          <p className="text-2xl font-bold text-blue-700">{inPlay}</p>
          {total > 0 && (
            <p className="text-xs text-gray-500">
              {((inPlay / total) * 100).toFixed(1)}%
            </p>
          )}
        </div>
        <div className="bg-gray-50 rounded-lg p-4">
          <p className="text-sm text-gray-600 mb-1">Total Shots</p>
          <p className="text-2xl font-bold text-gray-700">{total}</p>
        </div>
      </div>

      {Object.keys(statsSummary).length > 0 && (
        <div className="mt-6">
          <h3 className="text-lg font-semibold mb-3">Additional Stats</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {Object.entries(statsSummary).map(([key, value]) => (
              <div key={key} className="bg-gray-50 rounded p-3">
                <p className="text-sm text-gray-600 capitalize mb-1">
                  {key.replace(/_/g, ' ')}
                </p>
                <p className="text-lg font-semibold">{String(value)}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
