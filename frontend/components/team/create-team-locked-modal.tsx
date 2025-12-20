"use client"

import { useRouter } from "next/navigation"
import { X, Key, Users } from "lucide-react"
import { Button } from "@/components/ui/button"

interface CreateTeamLockedModalProps {
  isOpen: boolean
  onClose: () => void
}

export function CreateTeamLockedModal({ isOpen, onClose }: CreateTeamLockedModalProps) {
  const router = useRouter()

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-[#1a1a1a] max-w-md w-full rounded-2xl border border-[#333333] shadow-2xl relative">
        <div className="flex justify-between items-center p-6 border-b border-[#333333]">
          <h2 className="text-xl font-bold text-white">Activation Required</h2>
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
              <Key className="h-8 w-8 text-yellow-400" />
            </div>
            <h3 className="text-lg font-semibold text-white mb-2">
              Activate Your Account to Create Teams
            </h3>
            <p className="text-gray-400 text-sm mb-4">
              To create a new team, you need to activate your account first. You can either:
            </p>
            <div className="space-y-2 text-left w-full">
              <div className="flex items-start gap-3 p-3 bg-black/50 rounded-lg">
                <Key className="h-5 w-5 text-[#50C878] flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-white text-sm font-medium">Enter an activation key</p>
                  <p className="text-gray-400 text-xs">If you've purchased access, enter your activation key above.</p>
                </div>
              </div>
              <div className="flex items-start gap-3 p-3 bg-black/50 rounded-lg">
                <Users className="h-5 w-5 text-[#50C878] flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-white text-sm font-medium">Join an existing team</p>
                  <p className="text-gray-400 text-xs">If another coach has already created a team, you can join it below.</p>
                </div>
              </div>
            </div>
          </div>

          <div className="flex gap-4">
            <Button
              variant="outline"
              onClick={onClose}
              className="flex-1 border-[#333333] text-white hover:bg-[#262626] hover:border-[#50C878]"
            >
              Got it
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
