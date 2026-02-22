'use client'

import { motion } from 'framer-motion'
import Image from 'next/image'
import { useTheme } from 'next-themes'
import { useEffect, useState } from 'react'

export const Overview = () => {
  const { theme, systemTheme } = useTheme()
  const [mounted, setMounted] = useState(false)

  // Avoid hydration mismatch
  useEffect(() => {
    setMounted(true)
  }, [])

  // Determine current theme
  const currentTheme = theme === 'system' ? systemTheme : theme
  const isDark = currentTheme === 'dark'

  // Select appropriate logo based on theme
  const logoSrc = mounted && isDark ? '/images/logo-white.png' : '/images/logo-color.png'

  return (
    <motion.div
      key="overview"
      className="max-w-3xl mx-auto md:mt-20"
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.98 }}
      transition={{ delay: 0.5 }}
    >
      <div className="rounded-xl p-6 flex flex-col gap-8 leading-relaxed text-center max-w-xl">
        {/* Logos */}
        <div className="flex flex-row justify-center gap-6 items-center">
          {/* Ohio University Logo - switches based on theme */}
          <div className="relative w-24 h-24 flex items-center justify-center">
            <Image
              src={logoSrc}
              alt="Ohio University"
              width={96}
              height={96}
              className="object-contain transition-opacity duration-300"
              priority
            />
          </div>
          
          <span className="text-3xl text-ohio-green dark:text-white font-bold">+</span>
          
          {/* Cranberry Logo */}
          <div className="relative w-24 h-24 flex items-center justify-center">
            <Image
              src="/images/cranberry.png"
              alt="Cranberry Research"
              width={96}
              height={96}
              className="object-contain"
              priority
            />
          </div>
        </div>

        {/* Title */}
        <div className="space-y-2">
          <h1 className="text-3xl font-bold text-ohio-green dark:text-white">
            Cranberry Chatbot
          </h1>
          <p className="text-lg text-ohio-dark dark:text-gray-300">
            Research Assistant powered by Ohio University
          </p>
        </div>

        {/* Optional: Add a subtle description */}
        <p className="text-sm text-muted-foreground max-w-md mx-auto">
          Ask questions about cranberry chemistry, analytics, and research. 
          All answers are grounded in our document corpus.
        </p>
      </div>
    </motion.div>
  )
}