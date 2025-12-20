'use client'

import { createClient } from '@/lib/supabase/client'
import { useRouter } from 'next/navigation'
import { useState } from 'react'

export function useAuth() {
  const [loading, setLoading] = useState(false)
  const router = useRouter()
  const supabase = createClient()

  const signUp = async (email: string, password: string, name?: string, role?: 'coach' | 'player') => {
    try {
      setLoading(true)
      const { data, error } = await supabase.auth.signUp({
        email,
        password,
        options: {
          data: {
            name: name || '',
            role: role || 'player', // Pass role to metadata for database trigger
          },
        },
      })

      if (error) throw error

      // User profile is automatically created by database trigger
      // No need to manually insert here
      
      // If session exists, email confirmations are disabled and user is auto-signed in
      // If no session, email confirmations are enabled and user needs to confirm email

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

      router.push('/')
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
