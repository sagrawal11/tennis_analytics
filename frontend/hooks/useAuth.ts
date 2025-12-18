'use client'

import { createClient } from '@/lib/supabase/client'
import { useRouter } from 'next/navigation'
import { useState } from 'react'

export function useAuth() {
  const [loading, setLoading] = useState(false)
  const router = useRouter()
  const supabase = createClient()

  const signUp = async (email: string, password: string, name?: string) => {
    try {
      setLoading(true)
      const { data, error } = await supabase.auth.signUp({
        email,
        password,
        options: {
          data: {
            name: name || '',
          },
        },
      })

      if (error) throw error

      // User profile is automatically created by database trigger
      // No need to manually insert here

      return { data, error: null }
    } catch (error: any) {
      return { data: null, error }
    } finally {
      setLoading(false)
    }
  }

  const signIn = async (email: string, password: string) => {
    try {
      setLoading(true)
      const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password,
      })

      if (error) throw error

      router.refresh()
      return { data, error: null }
    } catch (error: any) {
      return { data: null, error }
    } finally {
      setLoading(false)
    }
  }

  const signOut = async () => {
    try {
      setLoading(true)
      const { error } = await supabase.auth.signOut()
      if (error) throw error

      router.push('/login')
      router.refresh()
    } catch (error: any) {
      console.error('Error signing out:', error)
    } finally {
      setLoading(false)
    }
  }

  const getUser = async () => {
    const {
      data: { user },
    } = await supabase.auth.getUser()
    return user
  }

  const getSession = async () => {
    const {
      data: { session },
    } = await supabase.auth.getSession()
    return session
  }

  return {
    signUp,
    signIn,
    signOut,
    getUser,
    getSession,
    loading,
  }
}
