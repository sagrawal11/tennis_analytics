import { createClient } from '@/lib/supabase/server'
import { redirect } from 'next/navigation'
import { MainLayout } from '@/components/layout/MainLayout'
import { TeamsContent } from '@/components/team/TeamsContent'

export default async function TeamsPage() {
  const supabase = await createClient()
  const {
    data: { user },
  } = await supabase.auth.getUser()

  if (!user) {
    redirect('/login')
  }

  // Get user profile
  const { data: profile } = await supabase
    .from('users')
    .select('*')
    .eq('id', user.id)
    .single()

  return (
    <MainLayout>
      <TeamsContent profile={profile} />
    </MainLayout>
  )
}
