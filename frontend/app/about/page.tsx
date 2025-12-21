"use client"

import { useState } from "react"
import { LandingNav } from "@/components/landing/landing-nav"
import { Footer } from "@/components/landing/footer"
import { Button } from "@/components/ui/button"
import { AuthModal } from "@/components/auth/auth-modal"
import { AnimatedAboutCard } from "@/components/about/animated-about-card"

const aboutSection = {
  imageUrl: "https://images.unsplash.com/photo-1551698618-1dfe5d97d256?w=800&h=600&fit=crop&q=80",
  imageAlt: "About Courtvision",
  content: "Your content will go here. You can edit this text in the aboutSection object."
}

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
    <main className="min-h-screen bg-black overflow-x-hidden">
      <LandingNav />
      
      {/* Hero Section */}
      <section className="pt-32 pb-1 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-5xl md:text-6xl lg:text-7xl font-bold text-white mb-6 animate-fade-in">
            About <span className="text-[#50C878]">Courtvision</span>
          </h1>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto animate-fade-in-delay">
            Revolutionizing tennis analytics through cutting-edge computer vision technology
          </p>
        </div>
      </section>

      {/* Main Content Section */}
      <AnimatedAboutCard section={aboutSection} />

      {/* CTA Section */}
      <section className="py-12 px-4">
        <div className="max-w-4xl mx-auto">
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

      <Footer />
      <AuthModal isOpen={authOpen} onClose={() => setAuthOpen(false)} initialMode={authMode} />
    </main>
  )
}
