"use client"

import { useRouter } from "next/navigation"
import { X, Users } from "lucide-react"
import { Button } from "@/components/ui/button"

interface LockedUploadModalProps {
  isOpen: boolean
  onClose: () => void
}

export function LockedUploadModal({ isOpen, onClose }: LockedUploadModalProps) {
  const router = useRouter()

  if (!isOpen) return null

  const handleGoToTeams = () => {
    onClose()
    router.push("/teams")
  }

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-[#1a1a1a] max-w-md w-full rounded-2xl border border-[#333333] shadow-2xl relative">
        <div className="flex justify-between items-center p-6 border-b border-[#333333]">
          <h2 className="text-xl font-bold text-white">Join a Team First</h2>
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            className="text-gray-400 hover:text-white hover:bg-[#262626]"
          >
            <X className="h-5 w-5" />
          </Button>
        </div>
        
        <div className="p-6">
          <div className="flex flex-col items-center text-center mb-6">
            <div className="bg-yellow-900/20 border-2 border-yellow-800 rounded-full p-4 mb-4">
              <Users className="h-8 w-8 text-yellow-400" />
            </div>
            <h3 className="text-lg font-semibold text-white mb-2">
              Join an Activated Team to Get Started
            </h3>
            <p className="text-gray-400 text-sm">
              You need to join a team with an activated coach before you can upload and analyze matches. Ask your coach for a team code. If your coach hasn&apos;t activated their account yet, they&apos;ll need to do that first.
            </p>
          </div>

          <div className="flex gap-4">
            <Button
              variant="outline"
              onClick={onClose}
              className="flex-1 border-[#333333] text-white hover:bg-[#262626] hover:border-[#50C878]"
            >
              Cancel
            </Button>
            <Button
              onClick={handleGoToTeams}
              className="flex-1 bg-[#50C878] hover:bg-[#45b069] text-black font-semibold"
            >
              Go to Teams
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
