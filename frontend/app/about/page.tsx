"use client"

import { useState } from "react"
import { LandingNav } from "@/components/landing/landing-nav"
import { Footer } from "@/components/landing/footer"
import { Button } from "@/components/ui/button"
import { AuthModal } from "@/components/auth/auth-modal"

export default function AboutPage() {
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
            About Courtvision
          </h1>
          
          <div className="space-y-8">
            <section className="bg-[#1a1a1a] rounded-xl p-8 border border-[#333333]">
              <h2 className="text-2xl font-semibold text-white mb-4">Our Mission</h2>
              <p className="text-gray-400 leading-relaxed">
                Courtvision is dedicated to revolutionizing tennis analytics through advanced computer vision
                technology. We believe that every player, from beginners to professionals, deserves access to
                detailed performance insights that can help them improve their game.
              </p>
            </section>

            <section className="bg-[#1a1a1a] rounded-xl p-8 border border-[#333333]">
              <h2 className="text-2xl font-semibold text-white mb-4">Technology</h2>
              <p className="text-gray-400 leading-relaxed mb-4">
                Our platform leverages cutting-edge computer vision models to automatically analyze tennis matches:
              </p>
              <ul className="list-disc list-inside text-gray-400 space-y-2 ml-4">
                <li>Player detection and tracking using YOLOv8 and advanced pose estimation</li>
                <li>Ball tracking with TrackNet and YOLO fusion for robust detection</li>
                <li>Shot classification using machine learning models</li>
                <li>Court detection with geometric validation for accurate positioning</li>
                <li>Bounce detection through trajectory analysis</li>
              </ul>
            </section>

            <section className="bg-[#1a1a1a] rounded-xl p-8 border border-[#333333]">
              <h2 className="text-2xl font-semibold text-white mb-4">For Coaches and Players</h2>
              <p className="text-gray-400 leading-relaxed">
                Courtvision is designed for both coaches and players. Coaches can manage entire teams, track
                all their players' matches, and provide data-driven feedback. Players can upload their own
                matches, track their progress over time, and identify specific areas for improvement.
              </p>
            </section>

            <section className="bg-[#1a1a1a] rounded-xl p-8 border border-[#333333]">
              <h2 className="text-2xl font-semibold text-white mb-4">Privacy & Security</h2>
              <p className="text-gray-400 leading-relaxed">
                We take your data privacy seriously. All video processing happens securely, and your match
                data is only accessible to you and your team. We use industry-standard encryption and follow
                best practices for data security.
              </p>
            </section>

            <section className="bg-[#1a1a1a] rounded-xl p-8 border border-[#333333] text-center mt-8">
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
            </section>
          </div>
        </div>
      </div>
      <Footer />
      <AuthModal isOpen={authOpen} onClose={() => setAuthOpen(false)} initialMode={authMode} />
    </main>
  )
}
