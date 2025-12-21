"use client"

import { useState } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { AuthModal } from "@/components/auth/auth-modal"

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
        {/* Subtle green glow overlay - stronger in middle, fades at edges */}
        <div className="absolute inset-0 bg-gradient-to-br from-[#50C878]/8 via-[#50C878]/5 to-[#50C878]/3 pointer-events-none" />
        {/* Fade green glow out at bottom to preserve black border */}
        <div className="absolute bottom-0 left-0 right-0 h-[200px] bg-gradient-to-b from-transparent to-black pointer-events-none" />

        {/* Content */}
        <div className="relative z-10 max-w-7xl mx-auto px-4 pt-16">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12 items-center">
            {/* Left Side - Video Card */}
            <div className="order-2 lg:order-1 animate-fade-in">
              <div className="bg-[#1a1a1a] rounded-xl p-5 border-2 border-[#50C878]/50 shadow-2xl transition-all duration-300 ease-in-out hover:border-[#50C878]/70 hover:shadow-[0_0_30px_rgba(80,200,120,0.4),0_0_60px_rgba(80,200,120,0.2)]" style={{ boxShadow: '0 0 20px rgba(80, 200, 120, 0.3), 0 0 40px rgba(80, 200, 120, 0.15)' }}>
                <div className="w-full h-[300px] md:h-[450px] lg:h-[500px] bg-black rounded-lg overflow-hidden transition-transform duration-300 ease-in-out hover:scale-[1.01]">
                  {/* Placeholder for video - will be replaced with actual video */}
                  <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-[#0a0a0a] to-[#1a1a1a]">
                    <p className="text-gray-500 text-sm">Video placeholder - will be replaced with looped video</p>
                  </div>
                  {/* When ready, replace the div above with:
                  <video
                    autoPlay
                    loop
                    muted
                    playsInline
                    className="w-full h-full object-cover"
                  >
                    <source src="/path-to-your-video.mp4" type="video/mp4" />
                  </video>
                  */}
                </div>
              </div>
            </div>

            {/* Right Side - Text and CTA */}
            <div className="order-1 lg:order-2 text-center lg:text-left lg:ml-8 lg:pl-8 animate-slide-in-right">
              <h1 className="text-5xl md:text-6xl lg:text-7xl font-bold text-white mb-8 text-balance transition-all duration-300">
                Elevate Your Tennis Game with <span className="text-[#50C878] transition-colors duration-300">Computer Vision Powered</span> Analytics
              </h1>
              <div className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start">
                <Link href="/how-it-works">
                  <Button
                    variant="outline"
                    size="lg"
                    className="!border-white !text-white hover:!text-[#50C878] hover:!border-[#50C878] !bg-transparent hover:!bg-transparent text-lg px-8 py-6 transition-all duration-200 ease-in-out transform hover:scale-105"
                  >
                    How it Works
                  </Button>
                </Link>
                <Button
                  onClick={() => setAuthOpen(true)}
                  size="lg"
                  className="bg-[#50C878] hover:bg-[#45b069] text-black font-semibold text-lg px-8 py-6 transition-all duration-200 ease-in-out transform hover:scale-105"
                >
                  Get Started
                </Button>
              </div>
            </div>
          </div>
        </div>
      </section>

      <AuthModal isOpen={authOpen} onClose={() => setAuthOpen(false)} initialMode="signup" />
    </>
  )
}
