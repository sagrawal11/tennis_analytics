'use client'

import { useState } from 'react'

interface Shot {
  id: string
  start_pos: { x: number; y: number }
  end_pos: { x: number; y: number }
  result: 'winner' | 'error' | 'in_play'
  video_timestamp?: number
}

interface ShotLineProps {
  shot: Shot
  courtWidth: number
  courtHeight: number
  onClick?: () => void
}

export function ShotLine({ shot, courtWidth, courtHeight, onClick }: ShotLineProps) {
  const [isHovered, setIsHovered] = useState(false)

  // Convert percentage positions to pixel coordinates
  const startX = (shot.start_pos.x / 100) * courtWidth
  const startY = (shot.start_pos.y / 100) * courtHeight
  const endX = (shot.end_pos.x / 100) * courtWidth
  const endY = (shot.end_pos.y / 100) * courtHeight

  // Color based on result
  const colors = {
    winner: '#22c55e', // green
    error: '#ef4444', // red
    in_play: '#3b82f6', // blue
  }

  const color = colors[shot.result] || colors.in_play
  const strokeWidth = isHovered ? 3 : 2

  return (
    <g
      onClick={onClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      style={{ cursor: 'pointer' }}
    >
      {/* Shot line */}
      <line
        x1={startX}
        y1={startY}
        x2={endX}
        y2={endY}
        stroke={color}
        strokeWidth={strokeWidth}
        strokeDasharray={shot.result === 'error' ? '5,5' : '0'}
        opacity={isHovered ? 1 : 0.7}
      />
      {/* Start point */}
      <circle
        cx={startX}
        cy={startY}
        r={isHovered ? 6 : 4}
        fill={color}
        opacity={isHovered ? 1 : 0.8}
      />
      {/* End point */}
      <circle
        cx={endX}
        cy={endY}
        r={isHovered ? 6 : 4}
        fill={color}
        opacity={isHovered ? 1 : 0.8}
      />
    </g>
  )
}
