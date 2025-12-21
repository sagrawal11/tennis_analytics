import type React from "react"
import type { Metadata } from "next"
import { Geist, Geist_Mono } from "next/font/google"
import { Analytics } from "@vercel/analytics/next"
import { Providers } from "./providers"
import "./globals.css"

const _geist = Geist({ subsets: ["latin"] })
const _geistMono = Geist_Mono({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "Courtvision",
  description: "Elevate your tennis game with computer vision powered shot tracking and performance analytics",
  generator: "CourtVision",
  icons: {
    icon: "/CourtVisionLogo.png",
    apple: "/CourtVisionLogo.png",
  },
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body className="font-sans antialiased bg-black text-white">
        <Providers>
          {children}
        </Providers>
        <Analytics />
      </body>
    </html>
  )
}
