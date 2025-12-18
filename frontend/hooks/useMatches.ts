'use client'

import { useQuery } from '@tanstack/react-query'
import { createClient } from '@/lib/supabase/client'

export function useMatches() {
  const supabase = createClient()

  return useQuery({
    queryKey: ['matches'],
    queryFn: async () => {
      const { data, error } = await supabase
        .from('matches')
        .select('*')
        .order('created_at', { ascending: false })

      if (error) throw error
      return data || []
    },
  })
}
