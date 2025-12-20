import type React from "react"
import { Sidebar } from "./sidebar"
import { FloatingActionButton } from "./floating-action-button"

interface MainLayoutProps {
  children: React.ReactNode
}

export function MainLayout({ children }: MainLayoutProps) {
  return (
    <div className="min-h-screen bg-black">
      <Sidebar />
      <main className="ml-64 min-h-screen">{children}</main>
      <FloatingActionButton />
    </div>
  )
}
