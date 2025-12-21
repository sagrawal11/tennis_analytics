"use client"

import { useRef } from "react"
import { motion, useInView, useScroll, useTransform } from "framer-motion"
import Image from "next/image"

interface StepData {
  step: number
  title: string
  description: string
  imageUrl: string
  imageAlt: string
}

interface AnimatedStepCardProps {
  step: StepData
  index: number
}

export function AnimatedStepCard({ step, index }: AnimatedStepCardProps) {
  const ref = useRef<HTMLDivElement>(null)
  const isInView = useInView(ref, { once: false, margin: "-100px" })
  
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start end", "end start"]
  })

  // Animate opacity and position based on scroll
  const opacity = useTransform(scrollYProgress, [0, 0.3, 0.7, 1], [0, 1, 1, 0])
  const y = useTransform(scrollYProgress, [0, 0.3, 0.7, 1], [100, 0, 0, -100])
  const scale = useTransform(scrollYProgress, [0, 0.3, 0.7, 1], [0.8, 1, 1, 0.8])

  // Determine if image should be on left or right (alternating)
  const imageOnLeft = index % 2 === 0

  return (
    <motion.div
      ref={ref}
      className="flex items-center justify-center px-4 py-12 lg:py-16"
      style={{ opacity, y, scale }}
    >
      <div className="w-full">
        <div className={`flex flex-col ${imageOnLeft ? 'lg:flex-row' : 'lg:flex-row-reverse'} items-center gap-6 lg:gap-8`}>
          {/* Image Card */}
          <motion.div
            className="flex-1 w-full"
            initial={{ opacity: 0, x: imageOnLeft ? -100 : 100 }}
            animate={isInView ? { opacity: 1, x: 0 } : { opacity: 0, x: imageOnLeft ? -100 : 100 }}
            transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
          >
            <div className="relative aspect-video rounded-xl overflow-hidden border-2 border-[#50C878]/30 shadow-xl shadow-[#50C878]/10 group">
              <div className="absolute inset-0 bg-gradient-to-br from-[#50C878]/10 to-transparent z-10" />
              <Image
                src={step.imageUrl}
                alt={step.imageAlt}
                fill
                className="object-cover transition-transform duration-700 group-hover:scale-110"
                sizes="(max-width: 768px) 100vw, 50vw"
              />
              <div className="absolute top-4 left-4 z-20">
                <div className="bg-[#50C878] text-black font-bold text-2xl px-4 py-2 rounded-lg shadow-lg">
                  {step.step}
                </div>
              </div>
              {/* Animated glow effect */}
              <div className="absolute inset-0 border-2 border-[#50C878] rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 blur-sm" />
            </div>
          </motion.div>

          {/* Text Card */}
          <motion.div
            className="flex-1 w-full"
            initial={{ opacity: 0, x: imageOnLeft ? 100 : -100 }}
            animate={isInView ? { opacity: 1, x: 0 } : { opacity: 0, x: imageOnLeft ? 100 : -100 }}
            transition={{ duration: 0.8, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
          >
            <div className="bg-gradient-to-br from-[#1a1a1a] to-[#0f0f0f] rounded-xl p-6 lg:p-8 border border-[#333333] shadow-xl relative overflow-hidden group">
              {/* Animated background gradient */}
              <div className="absolute inset-0 bg-gradient-to-br from-[#50C878]/5 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-700" />
              
              {/* Decorative corner accent */}
              <div className="absolute top-0 right-0 w-32 h-32 bg-[#50C878]/10 rounded-bl-full blur-3xl" />
              
              <div className="relative z-10">
                <motion.h2
                  className="text-2xl lg:text-3xl font-bold text-white mb-4"
                  initial={{ opacity: 0, y: 20 }}
                  animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
                  transition={{ duration: 0.6, delay: 0.4 }}
                >
                  <span className="text-[#50C878]">{step.step}.</span> {step.title}
                </motion.h2>
                <motion.p
                  className="text-gray-300 text-base lg:text-lg leading-relaxed"
                  initial={{ opacity: 0, y: 20 }}
                  animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
                  transition={{ duration: 0.6, delay: 0.6 }}
                >
                  {step.description}
                </motion.p>
              </div>

              {/* Animated border glow on hover */}
              <div className="absolute inset-0 rounded-xl border-2 border-[#50C878] opacity-0 group-hover:opacity-30 transition-opacity duration-500" />
            </div>
          </motion.div>
        </div>
      </div>
    </motion.div>
  )
}
