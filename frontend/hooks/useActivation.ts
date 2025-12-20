'use client'

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { createClient } from '@/lib/supabase/client'
import { useAuth } from './useAuth'

export function useActivation() {
  const { getUser } = useAuth()
  const supabase = createClient()
  const queryClient = useQueryClient()

  const { data: status, isLoading } = useQuery({
    queryKey: ['activation-status'],
    queryFn: async () => {
      const user = await getUser()
      if (!user) return null

      const { data: { session } } = await supabase.auth.getSession()
      if (!session) return null

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/activation/status`, {
        headers: {
          'Authorization': `Bearer ${session.access_token}`,
        },
      })

      if (!response.ok) return null

      const data = await response.json()
      return data
    },
  })

  const activateMutation = useMutation({
    mutationFn: async (activationKey: string) => {
      const user = await getUser()
      if (!user) throw new Error('Not authenticated')

      const { data: { session } } = await supabase.auth.getSession()
      if (!session) throw new Error('Not authenticated')

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/activation/activate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${session.access_token}`,
        },
        body: JSON.stringify({ activation_key: activationKey }),
      })

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || 'Failed to activate account')
      }

      return response.json()
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['activation-status'] })
      queryClient.invalidateQueries({ queryKey: ['profile'] })
    },
  })

  return {
    status,
    isActivated: status?.is_activated || false,
    isLoading,
    activate: activateMutation.mutateAsync,
    isActivating: activateMutation.isPending,
    activationError: activateMutation.error,
  }
}
