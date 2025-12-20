"use client"

import { useState } from "react"
import { LandingNav } from "@/components/landing/landing-nav"
import { Footer } from "@/components/landing/footer"
import { Button } from "@/components/ui/button"
import { AuthModal } from "@/components/auth/auth-modal"
import { Check } from "lucide-react"

export default function PricingPage() {
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

  const plans = [
    {
      name: "Individual",
      price: "$20/month",
      period: "",
      description: "Perfect for individual players and coaches",
      features: [
        "Unlimited match uploads",
        "Advanced shot tracking and visualization",
        "Personal statistics dashboard",
        "Interactive court diagram",
        "Video playback with shot timestamps",
        "Team management (up to 20 players)",
        "Player comparison tools",
        "Export statistics to PDF/CSV",
        "Priority processing",
        "Email support",
      ],
      cta: "Get Started",
      popular: true,
    },
    {
      name: "Team",
      price: "Custom",
      period: "Pricing",
      description: "For teams and organizations",
      features: [
        "Everything in Individual",
        "Unlimited team members",
        "Advanced team analytics",
        "Custom reporting",
        "API access",
        "Dedicated account manager",
        "Custom integrations",
        "Priority support",
      ],
      cta: "Contact Sales",
      popular: false,
    },
  ]

  return (
    <main className="min-h-screen bg-black">
      <LandingNav />
      <div className="pt-16 pb-24">
        <div className="max-w-7xl mx-auto px-4 py-16">
          <div className="text-center mb-16">
            <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">
              Simple, Transparent Pricing
            </h1>
            <p className="text-xl text-gray-400 max-w-2xl mx-auto">
              Choose the plan that's right for you. All plans include our core computer vision powered analytics.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-16 max-w-4xl mx-auto">
            {plans.map((plan) => (
              <div
                key={plan.name}
                className={`bg-[#1a1a1a] rounded-xl p-8 border ${
                  plan.popular
                    ? "border-[#50C878] shadow-lg shadow-[#50C878]/20"
                    : "border-[#333333]"
                } relative`}
              >
                {plan.popular && (
                  <div className="absolute -top-4 left-1/2 -translate-x-1/2">
                    <span className="bg-[#50C878] text-black text-sm font-semibold px-4 py-1 rounded-full">
                      Most Popular
                    </span>
                  </div>
                )}
                
                <div className="mb-6">
                  <h3 className="text-2xl font-bold text-white mb-2">{plan.name}</h3>
                  <div className="flex items-baseline gap-2 mb-2">
                    <span className="text-4xl font-bold text-white">{plan.price}</span>
                    {plan.period && plan.period !== "Pricing" && (
                      <span className="text-gray-400">/{plan.period}</span>
                    )}
                  </div>
                  <p className="text-gray-400 text-sm">{plan.description}</p>
                </div>

                <ul className="space-y-4 mb-8">
                  {plan.features.map((feature, index) => (
                    <li key={index} className="flex items-start gap-3">
                      <Check className="h-5 w-5 text-[#50C878] flex-shrink-0 mt-0.5" />
                      <span className="text-gray-300 text-sm">{feature}</span>
                    </li>
                  ))}
                </ul>

                <Button
                  onClick={plan.name === "Team" ? () => {} : openSignUp}
                  className={`w-full ${
                    plan.popular
                      ? "bg-[#50C878] hover:bg-[#45b069] text-black"
                      : "bg-[#262626] hover:bg-[#333333] text-white border border-[#333333]"
                  } font-semibold`}
                >
                  {plan.cta}
                </Button>
              </div>
            ))}
          </div>

          <div className="bg-[#1a1a1a] rounded-xl p-8 border border-[#333333] text-center">
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
