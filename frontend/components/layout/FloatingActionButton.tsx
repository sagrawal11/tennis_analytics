'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Plus } from 'lucide-react'
import { UploadModal } from '@/components/upload/UploadModal'

export function FloatingActionButton() {
  const [isModalOpen, setIsModalOpen] = useState(false)

  return (
    <>
      <div className="fixed bottom-6 right-6 z-40">
        <Button
          onClick={() => setIsModalOpen(true)}
          size="lg"
          className="h-14 w-14 rounded-full shadow-lg"
        >
          <Plus className="h-6 w-6" />
        </Button>
      </div>
      <UploadModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} />
    </>
  )
}
