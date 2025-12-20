'use client'

import { useQuery } from '@tanstack/react-query'
import { createClient } from '@/lib/supabase/client'

export function useProfile() {
  const supabase = createClient()

  const { data: profile, isLoading, error } = useQuery({
    queryKey: ['profile'],
    queryFn: async () => {
      const {
        data: { user },
      } = await supabase.auth.getUser()
      if (!user) return null

      const { data, error } = await supabase.from('users').select('*').eq('id', user.id).single()

      if (error) return null
      return data
    },
  })

  return {
    profile,
    isLoading,
    error,
  }
}
