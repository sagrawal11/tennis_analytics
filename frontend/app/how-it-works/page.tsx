"use client"

import { useState } from "react"
import { LandingNav } from "@/components/landing/landing-nav"
import { Footer } from "@/components/landing/footer"
import { Button } from "@/components/ui/button"
import { AuthModal } from "@/components/auth/auth-modal"

export default function HowItWorksPage() {
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
    <main className="min-h-screen bg-black">
      <LandingNav />
      <div className="pt-16 pb-24">
        <div className="max-w-4xl mx-auto px-4 py-16">
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-8 text-center">
            How It Works
          </h1>
          
          <div className="space-y-12">
            <section className="bg-[#1a1a1a] rounded-xl p-8 border border-[#333333]">
              <h2 className="text-2xl font-semibold text-white mb-4">1. Upload Your Match</h2>
              <p className="text-gray-400 leading-relaxed">
                Simply paste your Playsight video link into our upload modal. Our system will extract the video
                and prepare it for analysis. No need to download or process files manually.
              </p>
            </section>

            <section className="bg-[#1a1a1a] rounded-xl p-8 border border-[#333333]">
              <h2 className="text-2xl font-semibold text-white mb-4">2. Identify the Player</h2>
              <p className="text-gray-400 leading-relaxed">
                We'll show you several frames from the video. Click on yourself (or the player you want to track)
                in each frame. Our computer vision system uses this information to track the player throughout
                the entire match using advanced color recognition and player tracking algorithms.
              </p>
            </section>

            <section className="bg-[#1a1a1a] rounded-xl p-8 border border-[#333333]">
              <h2 className="text-2xl font-semibold text-white mb-4">3. Automatic Processing</h2>
              <p className="text-gray-400 leading-relaxed">
                Our computer vision backend processes the video automatically. Using state-of-the-art models
                for player detection, ball tracking, and shot classification, we analyze every moment of the
                match. This typically takes about an hour, and you'll receive real-time updates on the progress.
              </p>
            </section>

            <section className="bg-[#1a1a1a] rounded-xl p-8 border border-[#333333]">
              <h2 className="text-2xl font-semibold text-white mb-4">4. View Your Analytics</h2>
              <p className="text-gray-400 leading-relaxed">
                Once processing is complete, explore your match on an interactive court diagram. Every shot
                is visualized with color-coded lines showing winners (green), errors (red), and shots in play
                (blue). Click on any shot to jump directly to that moment in the video. View comprehensive
                statistics broken down by game, set, and match.
              </p>
            </section>

            <section className="bg-[#1a1a1a] rounded-xl p-8 border border-[#333333]">
              <h2 className="text-2xl font-semibold text-white mb-4">5. Track Your Progress</h2>
              <p className="text-gray-400 leading-relaxed">
                Coaches can view all their players' matches in one dashboard, with powerful filtering options.
                Players can track their own performance over time. Our statistics page provides detailed
                insights into your game patterns, helping you identify areas for improvement.
              </p>
            </section>
          </div>

          <div className="mt-12 bg-[#1a1a1a] rounded-xl p-8 border border-[#333333] text-center">
            <h2 className="text-2xl font-semibold text-white mb-4">
              Ready to get started?
            </h2>
            <p className="text-gray-400 mb-6">
              Sign in to your existing account or create a new one to start analyzing your matches today.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button
                onClick={openSignIn}
                variant="outline"
                className="!border-white !text-white hover:!text-[#50C878] hover:!border-[#50C878] !bg-transparent hover:!bg-transparent"
              >
                Sign In
              </Button>
              <Button
                onClick={openSignUp}
                className="bg-[#50C878] hover:bg-[#45b069] text-black font-semibold"
              >
                Create Account
              </Button>
            </div>
          </div>
        </div>
      </div>
      <Footer />
      <AuthModal isOpen={authOpen} onClose={() => setAuthOpen(false)} initialMode={authMode} />
    </main>
  )
}
