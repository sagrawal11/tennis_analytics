"use client"

import { useState, useCallback } from "react"
import { createBrowserClient } from "@supabase/ssr"

const supabase = createBrowserClient(process.env.NEXT_PUBLIC_SUPABASE_URL!, process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!)

export function useAuth() {
  const [loading, setLoading] = useState(false)

  const signUp = useCallback(
    async (email: string, password: string, name?: string, role: "coach" | "player" = "player") => {
      setLoading(true)
      try {
        const { data, error } = await supabase.auth.signUp({
          email,
          password,
          options: {
            emailRedirectTo: process.env.NEXT_PUBLIC_DEV_SUPABASE_REDIRECT_URL || `${window.location.origin}/dashboard`,
            data: {
              name,
              role,
            },
          },
        })
        return { data, error }
      } finally {
        setLoading(false)
      }
    },
    [],
  )

  const signIn = useCallback(async (email: string, password: string) => {
    setLoading(true)
    try {
      const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password,
      })
      return { data, error }
    } finally {
      setLoading(false)
    }
  }, [])

  const signOut = useCallback(async () => {
    setLoading(true)
    try {
      await supabase.auth.signOut()
    } finally {
      setLoading(false)
    }
  }, [])

  const getUser = useCallback(async () => {
    const {
      data: { user },
    } = await supabase.auth.getUser()
    return user
  }, [])

  const getSession = useCallback(async () => {
    const {
      data: { session },
    } = await supabase.auth.getSession()
    return session
  }, [])

  return {
    signUp,
    signIn,
    signOut,
    getUser,
    getSession,
    loading,
  }
}
