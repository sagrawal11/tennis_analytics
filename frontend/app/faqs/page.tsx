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
                  <p className="text-white">Ask your coach for your team code, then go to the Teams page and enter the code to join. Note: As a player, you&apos;ll only be able to upload videos once your team has an activated coach.</p>
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
              <h2 className="text-2xl font-semibold text-white mb-4">Account & Activation</h2>
              <div className="space-y-4">
                <div>
                  <h3 className="text-lg font-medium text-[#50C878] mb-2">What is an activation key?</h3>
                  <p className="text-white">Activation keys are provided to coaches after payment. Coaches need to enter their activation key to unlock all features, including the ability to create teams and upload matches.</p>
                </div>
                <div>
                  <h3 className="text-lg font-medium text-[#50C878] mb-2">Do I need an activation key as a coach?</h3>
                  <p className="text-white">Yes, coaches need an activation key to create teams. However, if another coach has already created a team and activated their account, you can join that team and your account will be automatically activated.</p>
                </div>
                <div>
                  <h3 className="text-lg font-medium text-[#50C878] mb-2">Why can&apos;t I upload videos as a player?</h3>
                  <p className="text-white">Players can only upload videos if they&apos;ve joined a team that has an activated coach. If your coach hasn&apos;t activated their account yet, ask them to enter their activation key or contact us for assistance.</p>
                </div>
                <div>
                  <h3 className="text-lg font-medium text-[#50C878] mb-2">What happens if I join a team with an activated coach?</h3>
                  <p className="text-white">If you&apos;re a coach and join a team that already has an activated coach, your account will be automatically activated. This allows multiple coaches to work together without each needing their own activation key.</p>
                </div>
                <div>
                  <h3 className="text-lg font-medium text-[#50C878] mb-2">How do I change my password?</h3>
                  <p className="text-white">Go to your profile page and use the password reset option. You&apos;ll receive an email with instructions.</p>
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
