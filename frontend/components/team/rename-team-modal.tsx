"use client"

import { useState } from "react"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { createClient } from "@/lib/supabase/client"
import { useQueryClient } from "@tanstack/react-query"

interface RenameTeamModalProps {
  isOpen: boolean
  onClose: () => void
  teamId: string
  currentName: string
}

export function RenameTeamModal({ isOpen, onClose, teamId, currentName }: RenameTeamModalProps) {
  const [newName1, setNewName1] = useState("")
  const [newName2, setNewName2] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const supabase = createClient()
  const queryClient = useQueryClient()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)

    // Validation
    if (!newName1.trim()) {
      setError("Please enter a team name")
      return
    }

    if (newName1.trim() !== newName2.trim()) {
      setError("Team names do not match. Please type the same name in both fields.")
      return
    }

    if (newName1.trim() === currentName) {
      setError("The new name must be different from the current name")
      return
    }

    setLoading(true)

    try {
      const { data: { session } } = await supabase.auth.getSession()
      if (!session) {
        setError("Please sign in first")
        setLoading(false)
        return
      }

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/teams/${teamId}/rename`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${session.access_token}`,
        },
        body: JSON.stringify({
          name: newName1.trim(),
        }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || data.message || 'Failed to rename team')
      }

      // Invalidate teams query to refresh the list
      queryClient.invalidateQueries({ queryKey: ['my-teams'] })

      // Reset form and close
      setNewName1("")
      setNewName2("")
      setError(null)
      onClose()
    } catch (err: unknown) {
      console.error('Rename error:', err)
      setError(err instanceof Error ? err.message : "Failed to rename team")
      setLoading(false)
    }
  }

  const handleClose = () => {
    if (!loading) {
      setNewName1("")
      setNewName2("")
      setError(null)
      onClose()
    }
  }

  return (
    <Dialog open={isOpen} onOpenChange={handleClose}>
      <DialogContent className="bg-[#1a1a1a] border-[#333333] text-white">
        <DialogHeader>
          <DialogTitle className="text-white">Rename Team</DialogTitle>
          <DialogDescription className="text-gray-400">
            Enter the new team name twice to confirm. This change will be visible to all team members.
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <Label htmlFor="newName1" className="text-gray-400 text-sm font-medium">
              New Team Name
            </Label>
            <Input
              id="newName1"
              type="text"
              value={newName1}
              onChange={(e) => setNewName1(e.target.value)}
              placeholder="Enter new team name"
              className="mt-1 bg-black/50 border-[#333333] text-white placeholder-gray-500 focus:border-[#50C878] focus:ring-[#50C878]"
              disabled={loading}
              required
            />
          </div>

          <div>
            <Label htmlFor="newName2" className="text-gray-400 text-sm font-medium">
              Confirm Team Name
            </Label>
            <Input
              id="newName2"
              type="text"
              value={newName2}
              onChange={(e) => setNewName2(e.target.value)}
              placeholder="Type the name again to confirm"
              className="mt-1 bg-black/50 border-[#333333] text-white placeholder-gray-500 focus:border-[#50C878] focus:ring-[#50C878]"
              disabled={loading}
              required
            />
          </div>

          {error && (
            <div className="bg-red-900/20 border border-red-800 rounded-lg p-3">
              <p className="text-sm text-red-300">{error}</p>
            </div>
          )}

          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={handleClose}
              disabled={loading}
              className="border-white text-white hover:border-red-600 hover:text-red-400 !bg-transparent hover:!bg-transparent"
            >
              Cancel
            </Button>
            <Button
              type="submit"
              disabled={loading}
              className="bg-[#50C878] hover:bg-[#45b069] text-black font-semibold"
            >
              {loading ? "Renaming..." : "Rename Team"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
