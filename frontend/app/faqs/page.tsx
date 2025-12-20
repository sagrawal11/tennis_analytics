"use client"

import { useState } from "react"
import { LandingNav } from "@/components/landing/landing-nav"
import { Footer } from "@/components/landing/footer"
import { Button } from "@/components/ui/button"
import { AuthModal } from "@/components/auth/auth-modal"

export default function FAQsPage() {
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
            Frequently Asked Questions
          </h1>
          
          <div className="space-y-8">
            <section className="bg-[#1a1a1a] rounded-xl p-8 border border-[#333333]">
              <h2 className="text-2xl font-semibold text-white mb-4">Getting Started</h2>
              <div className="space-y-4">
                <div>
                  <h3 className="text-lg font-medium text-[#50C878] mb-2">How do I create an account?</h3>
                  <p className="text-white">Click "Sign Up" in the top right corner, select your role (Coach or Player), fill in your information, and you're ready to go!</p>
                </div>
                <div>
                  <h3 className="text-lg font-medium text-[#50C878] mb-2">How do I join a team?</h3>
                  <p className="text-white">Ask your coach for your team code, then go to the Teams page and enter the code to join.</p>
                </div>
                <div>
                  <h3 className="text-lg font-medium text-[#50C878] mb-2">How do I upload a match?</h3>
                  <p className="text-white">Click the "+" button in the bottom right corner, enter your Playsight video link, and follow the prompts to identify yourself in the video frames.</p>
                </div>
              </div>
            </section>

            <section className="bg-[#1a1a1a] rounded-xl p-8 border border-[#333333]">
              <h2 className="text-2xl font-semibold text-white mb-4">Video Processing</h2>
              <div className="space-y-4">
                <div>
                  <h3 className="text-lg font-medium text-[#50C878] mb-2">How long does processing take?</h3>
                  <p className="text-white">Video processing typically takes about an hour, depending on the length of your match. You'll receive real-time updates on the progress.</p>
                </div>
                <div>
                  <h3 className="text-lg font-medium text-[#50C878] mb-2">What video formats are supported?</h3>
                  <p className="text-white">Currently, we support Playsight video links. Simply paste your Playsight link when uploading a match.</p>
                </div>
                <div>
                  <h3 className="text-lg font-medium text-[#50C878] mb-2">What if my video fails to process?</h3>
                  <p className="text-white">If processing fails, please contact our support team. We'll help you troubleshoot the issue.</p>
                </div>
              </div>
            </section>

            <section className="bg-[#1a1a1a] rounded-xl p-8 border border-[#333333]">
              <h2 className="text-2xl font-semibold text-white mb-4">Account & Billing</h2>
              <div className="space-y-4">
                <div>
                  <h3 className="text-lg font-medium text-[#50C878] mb-2">How do I change my password?</h3>
                  <p className="text-white">Go to your profile page and use the password reset option. You'll receive an email with instructions.</p>
                </div>
                <div>
                  <h3 className="text-lg font-medium text-[#50C878] mb-2">Can I cancel my subscription?</h3>
                  <p className="text-white">Yes, you can manage your subscription from your profile page at any time.</p>
                </div>
              </div>
            </section>

            <section className="bg-[#1a1a1a] rounded-xl p-8 border border-[#333333]">
              <h2 className="text-2xl font-semibold text-white mb-4">Still Need Help?</h2>
              <p className="text-gray-400 mb-4">
                If you can't find what you're looking for, feel free to contact us using the contact form in the footer.
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
