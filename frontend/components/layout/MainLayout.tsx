'use client'

import { Sidebar } from './Sidebar'
import { FloatingActionButton } from './FloatingActionButton'

export function MainLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex-1 overflow-y-auto bg-gray-50">
        <div className="relative h-full">
          {children}
          <FloatingActionButton />
        </div>
      </main>
    </div>
  )
}
