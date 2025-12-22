"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { useAuth } from "@/hooks/useAuth"

interface AuthModalProps {
  isOpen: boolean
  onClose: () => void
  initialMode?: "signin" | "signup"
}

export function AuthModal({ isOpen, onClose, initialMode = "signin" }: AuthModalProps) {
  const router = useRouter()
  const { signIn, signUp, loading } = useAuth()

  const [isSignUp, setIsSignUp] = useState(initialMode === "signup")
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [name, setName] = useState("")
  const [role, setRole] = useState<"player" | "coach">("player")
  const [error, setError] = useState<string | null>(null)

  // Password validation checks
  const passwordChecks = {
    hasLowercase: /[a-z]/.test(password),
    hasUppercase: /[A-Z]/.test(password),
    hasNumber: /[0-9]/.test(password),
    hasMinLength: password.length >= 8,
  }

  const isPasswordValid = Object.values(passwordChecks).every(Boolean)

  // Update isSignUp when initialMode changes (e.g., when switching between Sign In and Sign Up buttons)
  // Also reset form fields when modal opens or mode changes
  useEffect(() => {
    setIsSignUp(initialMode === "signup")
    if (isOpen) {
      setEmail("")
      setPassword("")
      setName("")
      setRole("player")
      setError(null)
    }
  }, [initialMode, isOpen])

  if (!isOpen) return null

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)

    // Validate password requirements for signup
    if (isSignUp && !isPasswordValid) {
      setError("Please meet all password requirements")
      return
    }

    try {
      if (isSignUp) {
        const { data, error } = await signUp(email, password, name, role)
        if (error) {
          // Filter out password validation errors - we show checklist instead
          if (error.message && error.message.toLowerCase().includes('password')) {
            // Only show non-password errors
            if (!error.message.toLowerCase().includes('must include')) {
              setError(error.message)
            }
          } else {
            setError(error.message)
          }
          return
        }
        if (data?.session) {
          onClose()
          router.push("/dashboard")
        } else {
          alert("Please check your email to confirm your account!")
          onClose()
        }
      } else {
        const { data, error } = await signIn(email, password)
        if (error) {
          setError(error.message)
          return
        }
        if (data?.session) {
          onClose()
          router.push("/dashboard")
        }
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "An error occurred")
    }
  }

  const toggleMode = () => {
    setIsSignUp(!isSignUp)
    setError(null)
  }

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-[#1a1a1a] max-w-md w-full rounded-2xl p-8 border border-[#333333] shadow-2xl">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-white">{isSignUp ? "Create account" : "Sign in"}</h2>
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            className="text-gray-400 hover:text-white hover:bg-[#262626]"
          >
            <X className="h-5 w-5" />
          </Button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {isSignUp && (
            <>
              <div>
                <Label htmlFor="name" className="text-gray-300 text-sm font-medium">
                  Full name
                </Label>
                <Input
                  id="name"
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Full name"
                  className="mt-1 bg-black/50 border-[#333333] text-white placeholder-gray-500 focus:border-[#50C878] focus:ring-[#50C878]"
                  required
                />
              </div>

              <div>
                <Label className="text-gray-300 text-sm font-medium block mb-2">I am a...</Label>
                <RadioGroup
                  value={role}
                  onValueChange={(value: "player" | "coach") => setRole(value)}
                  className="flex gap-4"
                >
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="player" id="player" className="border-[#333333] text-[#50C878]" />
                    <Label htmlFor="player" className="text-white cursor-pointer">
                      Player
                    </Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="coach" id="coach" className="border-[#333333] text-[#50C878]" />
                    <Label htmlFor="coach" className="text-white cursor-pointer">
                      Coach
                    </Label>
                  </div>
                </RadioGroup>
              </div>
            </>
          )}

          <div>
            <Label htmlFor="email" className="text-gray-300 text-sm font-medium">
              Email address
            </Label>
            <Input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Email address"
              autoComplete="email"
              className="mt-1 bg-black/50 border-[#333333] text-white placeholder-gray-500 focus:border-[#50C878] focus:ring-[#50C878]"
              required
            />
          </div>

          <div>
            <Label htmlFor="password" className="text-gray-300 text-sm font-medium">
              Password
            </Label>
            <Input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Password"
              autoComplete={isSignUp ? "new-password" : "current-password"}
              className="mt-1 bg-black/50 border-[#333333] text-white placeholder-gray-500 focus:border-[#50C878] focus:ring-[#50C878]"
              required
            />
            {isSignUp && (
              <div className="mt-2 space-y-1">
                <div className={`text-xs flex items-center gap-2 ${passwordChecks.hasLowercase ? 'text-[#50C878]' : 'text-red-400'}`}>
                  <span>{passwordChecks.hasLowercase ? '✓' : '•'}</span>
                  <span>Lowercase letter</span>
                </div>
                <div className={`text-xs flex items-center gap-2 ${passwordChecks.hasUppercase ? 'text-[#50C878]' : 'text-red-400'}`}>
                  <span>{passwordChecks.hasUppercase ? '✓' : '•'}</span>
                  <span>Uppercase letter</span>
                </div>
                <div className={`text-xs flex items-center gap-2 ${passwordChecks.hasNumber ? 'text-[#50C878]' : 'text-red-400'}`}>
                  <span>{passwordChecks.hasNumber ? '✓' : '•'}</span>
                  <span>Number</span>
                </div>
                <div className={`text-xs flex items-center gap-2 ${passwordChecks.hasMinLength ? 'text-[#50C878]' : 'text-red-400'}`}>
                  <span>{passwordChecks.hasMinLength ? '✓' : '•'}</span>
                  <span>8 characters</span>
                </div>
              </div>
            )}
          </div>

          {error && (
            <div className="bg-red-900/20 border border-red-800 rounded-lg p-3">
              <p className="text-sm text-red-300">{error}</p>
            </div>
          )}

          <Button
            type="submit"
            disabled={loading || (isSignUp && !isPasswordValid)}
            className="w-full bg-[#50C878] hover:bg-[#45b069] text-black font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? "Loading..." : isSignUp ? "Sign up" : "Sign in"}
          </Button>

          <button
            type="button"
            onClick={toggleMode}
            className="w-full text-center text-[#50C878] hover:text-[#6fd999] text-sm cursor-pointer"
          >
            {isSignUp ? "Already have an account? Sign in" : "Don't have an account? Sign up"}
          </button>
        </form>
      </div>
    </div>
  )
}
