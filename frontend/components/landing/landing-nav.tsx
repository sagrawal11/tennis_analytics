"use client"

import { useState } from "react"
import Image from "next/image"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { AuthModal } from "@/components/auth/auth-modal"

export function LandingNav() {
  const [authOpen, setAuthOpen] = useState(false)
  const [authMode, setAuthMode] = useState<"signin" | "signup">("signin")

  const openSignIn = () => {
    setAuthMode("signin")
    setAuthOpen(true)
  }

  const openSignUp = () => {
    setAuthMode("signup")
    setAuthOpen(true)
  }

  return (
    <>
      <nav className="fixed top-0 left-0 right-0 z-50 bg-black/80 backdrop-blur-md border-b border-[#333333]/50">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-3 hover:opacity-90 transition-opacity">
            <Image src="/CourtVisionLogo.png" alt="Courtvision Logo" width={40} height={40} className="w-10 h-10" />
            <span className="text-xl font-bold text-white">Courtvision</span>
          </Link>

          <div className="flex items-center gap-6">
            <Link href="/how-it-works">
              <Button
                variant="ghost"
                className="text-white hover:text-[#50C878] hover:bg-transparent"
              >
                How it Works
              </Button>
            </Link>
            <Link href="/about">
              <Button
                variant="ghost"
                className="text-white hover:text-[#50C878] hover:bg-transparent"
              >
                About
              </Button>
            </Link>
          </div>

          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              onClick={openSignIn}
              className="text-white hover:text-[#50C878] hover:bg-transparent"
            >
              Sign In
            </Button>
            <Button onClick={openSignUp} className="bg-[#50C878] hover:bg-[#45b069] text-black font-semibold">
              Sign Up
            </Button>
          </div>
        </div>
      </nav>

      <AuthModal isOpen={authOpen} onClose={() => setAuthOpen(false)} initialMode={authMode} />
    </>
  )
}
