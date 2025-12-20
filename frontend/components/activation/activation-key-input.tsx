"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useActivation } from "@/hooks/useActivation"
import { CheckCircle2, XCircle } from "lucide-react"

export function ActivationKeyInput() {
  const { isActivated, activate, isActivating, activationError } = useActivation()
  const [key, setKey] = useState("")
  const [error, setError] = useState<string | null>(null)

  const handleActivate = async () => {
    if (!key.trim()) {
      setError("Please enter an activation key")
      return
    }

    setError(null)

    try {
      await activate(key.trim().toUpperCase())
      setKey("")
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to activate account")
    }
  }

  if (isActivated) {
    return (
      <div className="bg-emerald-900/20 border-2 border-emerald-800 rounded-lg p-4 flex items-center gap-3">
        <CheckCircle2 className="h-5 w-5 text-emerald-400 flex-shrink-0" />
        <div>
          <p className="text-emerald-400 font-medium">Account Activated</p>
          <p className="text-sm text-gray-400">Your account is active and ready to use.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-[#1a1a1a] rounded-xl p-6 border border-[#333333] shadow-xl">
      <div className="flex items-center gap-3 mb-4">
        <XCircle className="h-5 w-5 text-yellow-400" />
        <h2 className="text-xl font-semibold text-white">Activation Required</h2>
      </div>
      <p className="text-gray-400 mb-6 text-sm">
        Enter your activation key to unlock all features. Contact us if you need an activation key.
      </p>
      
      <div className="space-y-4">
        <div>
          <Label htmlFor="activationKey" className="text-gray-300 text-sm font-medium">
            Activation Key
          </Label>
          <Input
            id="activationKey"
            type="text"
            value={key}
            onChange={(e) => setKey(e.target.value.toUpperCase())}
            placeholder="Enter activation key"
            className="mt-1 bg-black/50 border-[#333333] text-white placeholder-gray-500 focus:border-[#50C878] focus:ring-[#50C878] uppercase"
            disabled={isActivating}
          />
        </div>

        {(error || activationError) && (
          <div className="bg-red-900/20 border border-red-800 rounded-lg p-3">
            <p className="text-sm text-red-300">{error || (activationError as Error)?.message || "Failed to activate"}</p>
          </div>
        )}

        <Button
          onClick={handleActivate}
          disabled={isActivating || !key.trim()}
          className="w-full bg-[#50C878] hover:bg-[#45b069] text-black font-semibold"
        >
          {isActivating ? "Activating..." : "Activate Account"}
        </Button>
      </div>
    </div>
  )
}
