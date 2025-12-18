import { createClient } from '@/lib/supabase/server'
import { redirect } from 'next/navigation'
import { MainLayout } from '@/components/layout/MainLayout'
import { DashboardContent } from '@/components/dashboard/DashboardContent'

export default async function DashboardPage() {
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
      <DashboardContent user={user} profile={profile} />
    </MainLayout>
  )
}
