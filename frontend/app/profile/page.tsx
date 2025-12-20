"use client"

import { MainLayout } from "@/components/layout/main-layout"
import { useProfile } from "@/hooks/useProfile"
import { useTeams } from "@/hooks/useTeams"
import { useAuth } from "@/hooks/useAuth"
import { useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import { Trash2, Archive, Edit, LogOut } from "lucide-react"

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
                  <div className="space-y-3">
                    {teams.map((team) => {
                      const isCoach = profile?.role === "coach"
                      return (
                        <div key={team.id} className="bg-black/50 rounded border border-[#333333] p-4">
                          <div className="flex justify-between items-start mb-3">
                            <div>
                              <p className="font-medium text-white">{team.name}</p>
                              <p className="text-xs text-gray-400 font-mono mt-1">{team.code}</p>
                            </div>
                          </div>
                          
                          {isCoach ? (
                            <div className="flex flex-wrap gap-2 mt-3 pt-3 border-t border-[#333333]">
                              <Button
                                variant="outline"
                                size="sm"
                                className="border-[#333333] text-gray-300 hover:border-[#50C878] hover:text-[#50C878] hover:bg-transparent bg-transparent"
                              >
                                <Edit className="h-4 w-4 mr-2" />
                                Rename
                              </Button>
                              <Button
                                variant="outline"
                                size="sm"
                                className="border-[#333333] text-gray-300 hover:border-yellow-600 hover:text-yellow-500 hover:bg-transparent bg-transparent"
                              >
                                <Archive className="h-4 w-4 mr-2" />
                                Archive
                              </Button>
                              <Button
                                variant="outline"
                                size="sm"
                                className="border-red-800 text-red-400 hover:bg-red-600 hover:text-black hover:border-red-600 bg-transparent"
                              >
                                <Trash2 className="h-4 w-4 mr-2" />
                                Delete
                              </Button>
                            </div>
                          ) : (
                            <div className="flex flex-wrap gap-2 mt-3 pt-3 border-t border-[#333333]">
                              <Button
                                variant="outline"
                                size="sm"
                                className="border-red-800 text-red-400 hover:bg-red-600 hover:text-black hover:border-red-600 bg-transparent"
                              >
                                <LogOut className="h-4 w-4 mr-2" />
                                Leave Team
                              </Button>
                            </div>
                          )}
                        </div>
                      )
                    })}
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
