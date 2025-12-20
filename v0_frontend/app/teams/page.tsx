"use client"

import { useState } from "react"
import { MainLayout } from "@/components/layout/main-layout"
import { CreateTeam } from "@/components/team/create-team"
import { TeamCode } from "@/components/team/team-code"
import { useTeams } from "@/hooks/useTeams"
import { useProfile } from "@/hooks/useProfile"
import { Button } from "@/components/ui/button"

export default function TeamsPage() {
  const { teams, isLoading } = useTeams()
  const { profile } = useProfile()
  const [showCreate, setShowCreate] = useState(false)

  const isCoach = profile?.role === "coach"

  return (
    <MainLayout>
      <div className="mx-auto px-4 py-8 max-w-7xl">
        <h1 className="text-3xl font-bold text-white mb-6">Teams</h1>

        {isCoach ? (
          <div className="space-y-6">
            {/* Create Team Section */}
            <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-6 shadow-xl">
              <h2 className="text-xl font-semibold text-white mb-4">Create Team</h2>

              {teams.length === 0 && !showCreate && (
                <div className="mb-4">
                  <p className="text-gray-400 mb-2">Welcome! Get started by creating your first team.</p>
                  <p className="text-sm text-gray-500">Teams help you organize and track your players&apos; matches.</p>
                </div>
              )}

              {!showCreate ? (
                <Button
                  onClick={() => setShowCreate(true)}
                  className="bg-[#50C878] hover:bg-[#45b069] text-black font-semibold"
                >
                  Create New Team
                </Button>
              ) : (
                <CreateTeam onCreated={() => setShowCreate(false)} />
              )}
            </div>

            {/* Your Teams Section */}
            {!isLoading && teams.length > 0 && (
              <div className="space-y-4">
                <h2 className="text-xl font-semibold text-white">Your Teams</h2>
                {teams.map((team) => (
                  <div key={team.id} className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-6 shadow-xl">
                    <div className="flex justify-between items-center mb-4">
                      <h3 className="text-lg font-semibold text-white">{team.name}</h3>
                      <span className="bg-black/50 border border-[#50C878]/50 rounded px-3 py-1 text-sm font-mono font-semibold text-[#50C878]">
                        {team.code}
                      </span>
                    </div>
                    <div>
                      <h4 className="text-sm font-medium text-gray-400 mb-2">Team Members</h4>
                      <p className="text-sm text-gray-500">Share the team code with players to join.</p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-6">
            {/* Join Team Section */}
            <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-6 shadow-xl">
              <h2 className="text-xl font-semibold text-white mb-4">Join Team</h2>

              {teams.length === 0 && (
                <div className="mb-4">
                  <p className="text-gray-400 mb-2">Join a team to get started!</p>
                  <p className="text-sm text-gray-500">Ask your coach for a team code to join.</p>
                </div>
              )}

              <TeamCode />
            </div>

            {/* Your Teams Section */}
            {!isLoading && teams.length > 0 && (
              <div className="space-y-4">
                <h2 className="text-lg font-semibold text-white">Your Teams</h2>
                <div className="space-y-2">
                  {teams.map((team) => (
                    <div key={team.id} className="p-3 bg-black/50 border border-[#333333] rounded-lg">
                      <span className="font-medium text-white">{team.name}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </MainLayout>
  )
}
