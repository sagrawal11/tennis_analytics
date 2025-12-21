"use client"

import { useRef } from "react"
import { motion, useInView } from "framer-motion"
import Image from "next/image"

interface AboutSection {
  imageUrl: string
  imageAlt: string
  content: string
}

interface AnimatedAboutCardProps {
  section: AboutSection
}

export function AnimatedAboutCard({ section }: AnimatedAboutCardProps) {
  const ref = useRef<HTMLDivElement>(null)
  const isInView = useInView(ref, { once: true, margin: "-50px" })

  return (
    <div ref={ref} className="flex justify-center px-4 py-8 lg:py-12">
      <div className="w-full max-w-7xl">
        <div className="flex flex-col lg:flex-row items-center gap-8 lg:gap-12">
          {/* Image Card */}
          <motion.div
            className="flex-1 w-full"
            initial={{ opacity: 0, x: -50 }}
            animate={isInView ? { opacity: 1, x: 0 } : { opacity: 0, x: -50 }}
            transition={{ duration: 0.6, ease: "easeOut" }}
          >
            <div className="relative aspect-video rounded-xl overflow-hidden border-2 border-[#50C878]/30 shadow-xl shadow-[#50C878]/10 group">
              <div className="absolute inset-0 bg-gradient-to-br from-[#50C878]/10 to-transparent z-10" />
              <Image
                src={section.imageUrl}
                alt={section.imageAlt}
                fill
                className="object-cover transition-transform duration-700 group-hover:scale-110"
                sizes="(max-width: 1024px) 100vw, 50vw"
              />
              {/* Animated glow effect */}
              <div className="absolute inset-0 border-2 border-[#50C878] rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 blur-sm" />
            </div>
          </motion.div>

          {/* Text Card */}
          <motion.div
            className="flex-1 w-full"
            initial={{ opacity: 0, x: 50 }}
            animate={isInView ? { opacity: 1, x: 0 } : { opacity: 0, x: 50 }}
            transition={{ duration: 0.6, delay: 0.1, ease: "easeOut" }}
          >
            <div className="bg-gradient-to-br from-[#1a1a1a] to-[#0f0f0f] rounded-xl p-8 lg:p-12 border border-[#333333] shadow-xl relative overflow-hidden group h-full min-h-[500px] flex items-center">
              {/* Animated background gradient */}
              <div className="absolute inset-0 bg-gradient-to-br from-[#50C878]/5 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-700" />
              
              {/* Decorative corner accent */}
              <div className="absolute top-0 right-0 w-32 h-32 bg-[#50C878]/10 rounded-bl-full blur-3xl" />
              
              <div className="relative z-10 w-full">
                <div className="text-gray-300 text-base lg:text-lg leading-relaxed whitespace-pre-line">
                  {section.content}
                </div>
              </div>

              {/* Animated border glow on hover */}
              <div className="absolute inset-0 rounded-xl border-2 border-[#50C878] opacity-0 group-hover:opacity-30 transition-opacity duration-500" />
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  )
}
