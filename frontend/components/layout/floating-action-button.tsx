"use client"

import { useState } from "react"
import { Plus, Lock } from "lucide-react"
import { Button } from "@/components/ui/button"
import { UploadModal } from "@/components/upload/upload-modal"
import { LockedUploadModal } from "@/components/upload/locked-upload-modal"
import { useProfile } from "@/hooks/useProfile"
import { useTeams } from "@/hooks/useTeams"
import { useActivation } from "@/hooks/useActivation"
import { useTeamActivation } from "@/hooks/useTeamActivation"

export function FloatingActionButton() {
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [isLockedModalOpen, setIsLockedModalOpen] = useState(false)
  const { profile } = useProfile()
  const { teams } = useTeams()
  const { isActivated } = useActivation()
  const { hasActivatedTeam } = useTeamActivation()

  const isCoach = profile?.role === "coach"
  const isPlayer = profile?.role === "player"
  const hasJoinedTeam = teams && teams.length > 0

  const handleClick = () => {
    if (isCoach && !isActivated) {
      // Coach not activated - show locked state (button should be disabled)
      return
    }
    if (isPlayer && !hasJoinedTeam) {
      // Player not in team - show locked modal
      setIsLockedModalOpen(true)
      return
    }
    if (isPlayer && hasJoinedTeam && !hasActivatedTeam) {
      // Player in team but team doesn't have activated coach - show locked modal
      setIsLockedModalOpen(true)
      return
    }
    // All good - open upload modal
    setIsModalOpen(true)
  }

  // Determine if button should be locked
  const isLocked = (isCoach && !isActivated) || (isPlayer && (!hasJoinedTeam || !hasActivatedTeam))

  return (
    <>
      <Button
        onClick={handleClick}
        disabled={isLocked}
        className={`fixed bottom-6 right-6 z-40 h-14 w-14 rounded-full shadow-lg hover:scale-110 transition-transform ${
          isLocked
            ? "bg-gray-600 hover:bg-gray-600 cursor-not-allowed opacity-50"
            : "bg-[#50C878] hover:bg-[#45b069] shadow-[#50C878]/30"
        }`}
      >
        {isLocked ? (
          <Lock className="h-6 w-6 text-white" />
        ) : (
          <Plus className="h-6 w-6 text-black" />
        )}
      </Button>

      <UploadModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} />
      <LockedUploadModal isOpen={isLockedModalOpen} onClose={() => setIsLockedModalOpen(false)} />
    </>
  )
}
