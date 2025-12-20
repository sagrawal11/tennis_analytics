"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { AuthModal } from "@/components/auth/auth-modal"
import { Play } from "lucide-react"

export function HeroSection() {
  const [authOpen, setAuthOpen] = useState(false)

  return (
    <>
      <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
        {/* Background */}
        <div
          className="absolute inset-0 bg-cover bg-center"
          style={{
            backgroundImage: `url('/tennis-court-aerial-view-dark-dramatic-lighting.jpg')`,
          }}
        />
        <div className="absolute inset-0 bg-gradient-to-b from-black/70 via-black/60 to-black" />

        {/* Content */}
        <div className="relative z-10 max-w-4xl mx-auto px-4 text-center pt-16">
          <h1 className="text-5xl md:text-6xl lg:text-7xl font-bold text-white mb-6 text-balance">
            Elevate Your Tennis Game with <span className="text-[#50C878]">AI-Powered</span> Analytics
          </h1>
          <p className="text-xl md:text-2xl text-gray-300 mb-8 text-pretty max-w-2xl mx-auto">
            Track every shot, analyze every match, improve every day. Transform your performance with intelligent
            insights.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button
              onClick={() => setAuthOpen(true)}
              size="lg"
              className="bg-[#50C878] hover:bg-[#45b069] text-black font-semibold text-lg px-8 py-6"
            >
              Get Started Free
            </Button>
            <Button
              variant="outline"
              size="lg"
              className="border-white/30 text-white hover:bg-white/10 hover:border-white/50 text-lg px-8 py-6 bg-transparent"
            >
              <Play className="mr-2 h-5 w-5" />
              Watch Demo
            </Button>
          </div>
        </div>

        {/* Scroll indicator */}
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 animate-bounce">
          <div className="w-6 h-10 rounded-full border-2 border-white/30 flex items-start justify-center p-2">
            <div className="w-1 h-3 bg-white/50 rounded-full" />
          </div>
        </div>
      </section>

      <AuthModal isOpen={authOpen} onClose={() => setAuthOpen(false)} initialMode="signup" />
    </>
  )
}
