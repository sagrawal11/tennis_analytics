"use client"

import useSWR from "swr"
import { createBrowserClient } from "@supabase/ssr"
import type { User } from "@/lib/types"

const supabase = createBrowserClient(process.env.NEXT_PUBLIC_SUPABASE_URL!, process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!)

async function fetchProfile(): Promise<User | null> {
  const {
    data: { user },
  } = await supabase.auth.getUser()
  if (!user) return null

  const { data, error } = await supabase.from("users").select("*").eq("id", user.id).single()

  if (error) return null
  return data
}

export function useProfile() {
  const {
    data: profile,
    isLoading,
    error,
    mutate,
  } = useSWR<User | null>("profile", fetchProfile, {
    revalidateOnFocus: false,
  })

  return {
    profile,
    isLoading,
    error,
    mutate,
  }
}
