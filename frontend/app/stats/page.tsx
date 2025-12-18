import { createClient } from '@/lib/supabase/server'
import { redirect } from 'next/navigation'
import { MainLayout } from '@/components/layout/MainLayout'

export default async function StatsPage() {
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

  const isCoach = profile?.role === 'coach'

  return (
    <MainLayout>
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold mb-6">Statistics</h1>
        <div className="bg-white rounded-lg shadow p-6">
          {isCoach ? (
            <div>
              <h2 className="text-xl font-semibold mb-4">Coach View</h2>
              <p className="text-gray-600">
                Select a player to view their season statistics.
              </p>
              {/* TODO: Add player selection interface */}
            </div>
          ) : (
            <div>
              <h2 className="text-xl font-semibold mb-4">Your Statistics</h2>
              <p className="text-gray-600">
                View your season statistics and per-game breakdowns.
              </p>
              {/* TODO: Add stats display */}
            </div>
          )}
        </div>
      </div>
    </MainLayout>
  )
}
