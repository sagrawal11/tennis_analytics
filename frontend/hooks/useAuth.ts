'use client'

import { createClient } from '@/lib/supabase/client'
import { useRouter } from 'next/navigation'
import { useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'

export function useAuth() {
  const [loading, setLoading] = useState(false)
  const router = useRouter()
  const supabase = createClient()
  const queryClient = useQueryClient()

  const signUp = async (email: string, password: string, name?: string, role?: 'coach' | 'player') => {
    try {
      setLoading(true)
      // Ensure role is explicitly set (default to 'player' if not provided)
      const userRole = role || 'player'
      
      console.log('Signing up with role:', userRole) // Debug log
      
      const { data, error } = await supabase.auth.signUp({
        email,
        password,
        options: {
          data: {
            name: name || '',
            role: userRole, // Pass role to metadata for database trigger
          },
        },
      })

      if (error) throw error

      // User profile is automatically created by database trigger
      // The trigger reads role from auth.users.raw_user_meta_data->>'role'
      // No need to manually insert here
      
      // If session exists, email confirmations are disabled and user is auto-signed in
      // If no session, email confirmations are enabled and user needs to confirm email

      return { data, error: null }
    } catch (error: any) {
      console.error('Signup error:', error)
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

      // Clear all cached queries to ensure fresh data for the new user
      queryClient.clear()
      
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

      // Clear all cached queries to prevent showing previous user's data
      queryClient.clear()

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
