import { createClient } from '@/lib/supabase/server'
import { redirect } from 'next/navigation'
import { MainLayout } from '@/components/layout/MainLayout'
import { MatchDetailContent } from '@/components/match/MatchDetailContent'

export default async function MatchDetailPage({
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

  // Get match data
  const { data: match } = await supabase
    .from('matches')
    .select('*')
    .eq('id', params.id)
    .single()

  if (!match) {
    redirect('/dashboard')
  }

  // Get match data (JSON from CV backend)
  const { data: matchData } = await supabase
    .from('match_data')
    .select('*')
    .eq('match_id', params.id)
    .single()

  // Get shots
  const { data: shots } = await supabase
    .from('shots')
    .select('*')
    .eq('match_id', params.id)
    .order('timestamp', { ascending: true })

  return (
    <MainLayout>
      <MatchDetailContent
        match={match}
        matchData={matchData}
        shots={shots || []}
      />
    </MainLayout>
  )
}
