import { createClient } from '@/lib/supabase/server'
import { redirect } from 'next/navigation'
import { MainLayout } from '@/components/layout/MainLayout'
import { PlayerIdentification } from '@/components/upload/PlayerIdentification'

export default async function PlayerIdentificationPage({
  params,
}: {
  params: { id: string }
}) {
  const supabase = await createClient()
  const {
    data: { user },
  } = await supabase.auth.getUser()

  if (!user) {
    redirect('/login')
  }

  // Get match
  const { data: match } = await supabase
    .from('matches')
    .select('*')
    .eq('id', params.id)
    .single()

  if (!match) {
    redirect('/dashboard')
  }

  // Verify ownership
  if (match.user_id !== user.id) {
    redirect('/dashboard')
  }

  return (
    <MainLayout>
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold mb-6">Identify Player</h1>
        <PlayerIdentification matchId={params.id} playsightLink={match.playsight_link} />
      </div>
    </MainLayout>
  )
}
