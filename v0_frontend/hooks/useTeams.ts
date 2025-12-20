"use client"

import { useState, useCallback } from "react"
import useSWR from "swr"
import { createBrowserClient } from "@supabase/ssr"
import type { Team } from "@/lib/types"

const supabase = createBrowserClient(process.env.NEXT_PUBLIC_SUPABASE_URL!, process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!)

async function fetchTeams(): Promise<Team[]> {
  const {
    data: { user },
  } = await supabase.auth.getUser()
  if (!user) return []

  const { data, error } = await supabase.from("teams").select("*").order("created_at", { ascending: false })

  if (error) throw error
  return data || []
}

export function useTeams() {
  const {
    data: teams = [],
    isLoading,
    mutate,
  } = useSWR<Team[]>("teams", fetchTeams, {
    revalidateOnFocus: false,
  })

  const [isCreating, setIsCreating] = useState(false)
  const [isJoining, setIsJoining] = useState(false)

  const createTeam = useCallback(
    async (name: string) => {
      setIsCreating(true)
      try {
        const code = Math.random().toString(36).substring(2, 8).toUpperCase()
        const { data, error } = await supabase.from("teams").insert([{ name, code }]).select().single()

        if (error) throw error
        await mutate()
        return data
      } finally {
        setIsCreating(false)
      }
    },
    [mutate],
  )

  const joinTeam = useCallback(
    async (code: string) => {
      setIsJoining(true)
      try {
        const { data: team, error: findError } = await supabase
          .from("teams")
          .select("*")
          .eq("code", code.toUpperCase())
          .single()

        if (findError) throw new Error("Team not found")

        const {
          data: { user },
        } = await supabase.auth.getUser()
        if (!user) throw new Error("Not authenticated")

        const { error: updateError } = await supabase.from("users").update({ team_id: team.id }).eq("id", user.id)

        if (updateError) throw updateError
        await mutate()
        return team
      } finally {
        setIsJoining(false)
      }
    },
    [mutate],
  )

  return {
    teams,
    isLoading,
    createTeam,
    joinTeam,
    isCreating,
    isJoining,
  }
}
