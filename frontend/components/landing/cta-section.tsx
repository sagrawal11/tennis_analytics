"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { AuthModal } from "@/components/auth/auth-modal"

export function CTASection() {
  const [authOpen, setAuthOpen] = useState(false)
  const [authMode, setAuthMode] = useState<"signin" | "signup">("signup")

  const openSignUp = () => {
    setAuthMode("signup")
    setAuthOpen(true)
  }

  const openSignIn = () => {
    setAuthMode("signin")
    setAuthOpen(true)
  }

  return (
    <>
      <section className="py-20 px-4 relative">
        {/* Base background */}
        <div className="absolute inset-0 bg-black" />
        {/* Pure black top half - no glow at all */}
        <div className="absolute top-0 left-0 right-0 h-1/2 bg-black pointer-events-none z-0" />
        {/* Subtle green glow - only in bottom half */}
        <div className="absolute bottom-0 left-0 right-0 h-1/2 bg-gradient-to-br from-[#50C878]/3 via-[#50C878]/4 to-[#50C878]/3 pointer-events-none z-0" />
        {/* Fade from pure black at top (200px) - ensures smooth transition */}
        <div className="absolute top-0 left-0 right-0 h-[200px] bg-gradient-to-b from-black to-transparent pointer-events-none z-5" />
        
        <div className="max-w-4xl mx-auto relative z-10">
          <div className="bg-gradient-to-br from-[#1a1a1a] to-[#0f0f0f] rounded-2xl p-8 lg:p-12 border border-[#333333] text-center relative overflow-hidden group">
            {/* Animated background */}
            <div className="absolute inset-0 bg-gradient-to-br from-[#50C878]/10 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-700" />
            <div className="absolute top-0 right-0 w-64 h-64 bg-[#50C878]/10 rounded-bl-full blur-3xl" />
            
            <div className="relative z-10">
              <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
                Ready to get started?
              </h2>
              <p className="text-gray-400 mb-8 text-lg">
                Sign in to your existing account or create a new one to start analyzing your matches today.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Button
                  onClick={openSignIn}
                  variant="outline"
                  className="!border-white !text-white hover:!text-[#50C878] hover:!border-[#50C878] !bg-transparent hover:!bg-transparent transition-all duration-200 ease-in-out transform hover:scale-105"
                >
                  Sign In
                </Button>
                <Button
                  onClick={openSignUp}
                  className="bg-[#50C878] hover:bg-[#45b069] text-black font-semibold transition-all duration-200 ease-in-out transform hover:scale-105"
                >
                  Create Account
                </Button>
              </div>
            </div>
          </div>
        </div>
      </section>

      <AuthModal isOpen={authOpen} onClose={() => setAuthOpen(false)} initialMode={authMode} />
    </>
  )
}
