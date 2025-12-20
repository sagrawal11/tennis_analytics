"use client"

import { useState, useEffect } from "react"
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from "@/components/ui/alert-dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

interface RemovePlayerModalProps {
  isOpen: boolean
  onClose: () => void
  onConfirm: () => void
  playerName: string
  teamName: string
  loading?: boolean
}

export function RemovePlayerModal({ isOpen, onClose, onConfirm, playerName, teamName, loading = false }: RemovePlayerModalProps) {
  const [confirmationText, setConfirmationText] = useState("")
  const requiredText = `Confirmation to remove ${playerName}`
  const isMatch = confirmationText === requiredText

  // Reset confirmation text when modal opens/closes
  useEffect(() => {
    if (!isOpen) {
      setConfirmationText("")
    }
  }, [isOpen])

  const handleConfirm = () => {
    if (isMatch && !loading) {
      onConfirm()
      setConfirmationText("")
    }
  }

  return (
    <AlertDialog open={isOpen} onOpenChange={onClose}>
      <AlertDialogContent className="bg-[#1a1a1a] border-[#333333] text-white">
        <AlertDialogHeader>
          <AlertDialogTitle className="text-white">Remove Player</AlertDialogTitle>
          <AlertDialogDescription className="text-gray-400">
            Are you sure you want to remove <span className="font-semibold text-white">{playerName}</span> from <span className="font-semibold text-white">{teamName}</span>?
            <br /><br />
            This will:
            <ul className="list-disc list-inside mt-2 space-y-1 text-sm">
              <li>Remove {playerName} from the team</li>
              <li>They will lose access to team matches and data</li>
              <li>They can rejoin later using the team code</li>
            </ul>
          </AlertDialogDescription>
        </AlertDialogHeader>

        <div className="space-y-2 mt-4">
          <Label htmlFor="confirmation" className="text-gray-400 text-sm font-medium">
            Type <span className="font-mono text-white">{requiredText}</span> to confirm:
          </Label>
          <Input
            id="confirmation"
            type="text"
            value={confirmationText}
            onChange={(e) => setConfirmationText(e.target.value)}
            placeholder={requiredText}
            className="bg-black/50 border-[#333333] text-white placeholder-gray-500 focus:border-red-600 focus:ring-red-600"
            disabled={loading}
            onKeyDown={(e) => {
              if (e.key === "Enter" && isMatch && !loading) {
                handleConfirm()
              }
            }}
          />
          {confirmationText && !isMatch && (
            <p className="text-xs text-red-400">Text does not match. Please type exactly: {requiredText}</p>
          )}
        </div>

        <AlertDialogFooter>
          <AlertDialogCancel
            onClick={onClose}
            disabled={loading}
            className="border-white text-white hover:border-red-600 hover:text-red-400 !bg-transparent hover:!bg-transparent"
          >
            Cancel
          </AlertDialogCancel>
          <AlertDialogAction
            onClick={handleConfirm}
            disabled={!isMatch || loading}
            className="bg-red-600 hover:bg-red-700 text-white disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? "Removing..." : "Remove Player"}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
