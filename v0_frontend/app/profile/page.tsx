"use client"

import { MainLayout } from "@/components/layout/main-layout"
import { useProfile } from "@/hooks/useProfile"
import { useTeams } from "@/hooks/useTeams"
import { useAuth } from "@/hooks/useAuth"
import { useEffect, useState } from "react"

export default function ProfilePage() {
  const { profile, isLoading } = useProfile()
  const { teams } = useTeams()
  const { getUser } = useAuth()
  const [email, setEmail] = useState<string | null>(null)

  useEffect(() => {
    const fetchEmail = async () => {
      const user = await getUser()
      setEmail(user?.email || null)
    }
    fetchEmail()
  }, [getUser])

  return (
    <MainLayout>
      <div className="mx-auto px-4 py-8 max-w-7xl">
        <h1 className="text-3xl font-bold text-white mb-6">Profile</h1>

        <div className="bg-[#1a1a1a] rounded-lg border border-[#333333] p-6 shadow-xl">
          {isLoading ? (
            <p className="text-gray-400">Loading profile...</p>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <p className="text-sm font-medium text-gray-400 mb-1">Name</p>
                <p className="text-base text-white">{profile?.name || "Not set"}</p>
              </div>

              <div>
                <p className="text-sm font-medium text-gray-400 mb-1">Email</p>
                <p className="text-base text-white">{email || "Loading..."}</p>
              </div>

              <div>
                <p className="text-sm font-medium text-gray-400 mb-1">Role</p>
                <span
                  className={`inline-flex px-2 py-1 rounded text-sm font-medium border ${
                    profile?.role === "coach"
                      ? "bg-blue-900/30 text-blue-400 border-blue-800"
                      : "bg-emerald-900/30 text-emerald-400 border-emerald-800"
                  }`}
                >
                  {profile?.role ? profile.role.charAt(0).toUpperCase() + profile.role.slice(1) : "Unknown"}
                </span>
              </div>

              <div>
                <p className="text-sm font-medium text-gray-400 mb-2">Teams</p>
                {teams.length === 0 ? (
                  <p className="text-sm text-gray-500 italic">Not part of any teams yet.</p>
                ) : (
                  <div className="space-y-2">
                    {teams.map((team) => (
                      <div key={team.id} className="bg-black/50 rounded border border-[#333333] p-3">
                        <p className="font-medium text-white">{team.name}</p>
                        <p className="text-xs text-gray-400 font-mono">{team.code}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </MainLayout>
  )
}
