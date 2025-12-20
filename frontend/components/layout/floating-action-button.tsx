"use client"

import { useState } from "react"
import { Plus } from "lucide-react"
import { Button } from "@/components/ui/button"
import { UploadModal } from "@/components/upload/upload-modal"

export function FloatingActionButton() {
  const [isModalOpen, setIsModalOpen] = useState(false)

  return (
    <>
      <Button
        onClick={() => setIsModalOpen(true)}
        className="fixed bottom-6 right-6 z-40 h-14 w-14 rounded-full bg-[#50C878] hover:bg-[#45b069] shadow-lg shadow-[#50C878]/30 hover:scale-110 transition-transform"
      >
        <Plus className="h-6 w-6 text-black" />
      </Button>

      <UploadModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} />
    </>
  )
}
