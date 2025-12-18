'use client'

import { useEffect, useState } from 'react'
import { createClient } from '@/lib/supabase/client'
import { useQuery } from '@tanstack/react-query'

interface ProcessingStatusProps {
  matchId: string
}

export function ProcessingStatus({ matchId }: ProcessingStatusProps) {
  const supabase = createClient()
  const [status, setStatus] = useState<string>('pending')

  // Poll for status updates
  const { data: matchStatus } = useQuery({
    queryKey: ['match-status', matchId],
    queryFn: async () => {
      const { data, error } = await supabase
        .from('matches')
        .select('status, processed_at')
        .eq('id', matchId)
        .single()

      if (error) throw error
      return data
    },
    refetchInterval: (data) => {
      // Poll every 5 seconds if still processing
      return data?.status === 'processing' ? 5000 : false
    },
  })

  useEffect(() => {
    if (matchStatus) {
      setStatus(matchStatus.status)
    }
  }, [matchStatus])

  // Subscribe to real-time updates
  useEffect(() => {
    const channel = supabase
      .channel(`match-${matchId}`)
      .on(
        'postgres_changes',
        {
          event: 'UPDATE',
          schema: 'public',
          table: 'matches',
          filter: `id=eq.${matchId}`,
        },
        (payload) => {
          const newStatus = payload.new.status
          setStatus(newStatus)
        }
      )
      .subscribe()

    return () => {
      supabase.removeChannel(channel)
    }
  }, [matchId, supabase])

  const statusConfig = {
    pending: {
      label: 'Pending',
      color: 'bg-yellow-100 text-yellow-800',
      message: 'Waiting to start processing...',
    },
    processing: {
      label: 'Processing',
      color: 'bg-blue-100 text-blue-800',
      message: 'Analyzing video... This may take up to an hour.',
    },
    completed: {
      label: 'Completed',
      color: 'bg-green-100 text-green-800',
      message: 'Processing complete! Your match is ready.',
    },
    failed: {
      label: 'Failed',
      color: 'bg-red-100 text-red-800',
      message: 'Processing failed. Please try again.',
    },
  }

  const config = statusConfig[status as keyof typeof statusConfig] || statusConfig.pending

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-center gap-4">
        <div className={`px-4 py-2 rounded-lg ${config.color}`}>
          <span className="font-semibold">{config.label}</span>
        </div>
        <div className="flex-1">
          <p className="text-sm text-gray-600">{config.message}</p>
          {matchStatus?.processed_at && (
            <p className="text-xs text-gray-500 mt-1">
              Completed: {new Date(matchStatus.processed_at).toLocaleString()}
            </p>
          )}
        </div>
        {status === 'processing' && (
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
        )}
      </div>
    </div>
  )
}
