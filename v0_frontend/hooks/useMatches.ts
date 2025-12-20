"use client"

import useSWR from "swr"
import { createBrowserClient } from "@supabase/ssr"
import type { Match } from "@/lib/types"

const supabase = createBrowserClient(process.env.NEXT_PUBLIC_SUPABASE_URL!, process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!)

async function fetchMatches(): Promise<Match[]> {
  const {
    data: { user },
  } = await supabase.auth.getUser()
  if (!user) return []

  const { data, error } = await supabase.from("matches").select("*").order("created_at", { ascending: false })

  if (error) throw error
  return data || []
}

export function useMatches() {
  const { data, error, isLoading, mutate } = useSWR<Match[]>("matches", fetchMatches, {
    revalidateOnFocus: false,
  })

  return {
    data,
    isLoading,
    error,
    mutate,
  }
}
