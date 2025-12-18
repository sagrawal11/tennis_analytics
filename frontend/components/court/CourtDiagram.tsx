'use client'

import { ShotLine } from './ShotLine'

interface Shot {
  id: string
  start_pos: { x: number; y: number }
  end_pos: { x: number; y: number }
  result: 'winner' | 'error' | 'in_play'
  video_timestamp?: number
}

interface CourtDiagramProps {
  shots?: Shot[]
  onShotClick?: (shot: Shot) => void
}

export function CourtDiagram({ shots = [], onShotClick }: CourtDiagramProps) {
  // Court dimensions (proportional)
  const courtWidth = 800
  const courtHeight = 400
  const serviceBoxWidth = courtWidth / 3
  const serviceBoxHeight = courtHeight / 2

  return (
    <div className="w-full bg-green-100 rounded-lg p-4 overflow-x-auto">
      <svg
        width={courtWidth}
        height={courtHeight}
        viewBox={`0 0 ${courtWidth} ${courtHeight}`}
        className="mx-auto"
      >
        {/* Court outline */}
        <rect
          x="0"
          y="0"
          width={courtWidth}
          height={courtHeight}
          fill="#4ade80"
          stroke="#22c55e"
          strokeWidth="2"
        />

        {/* Net line */}
        <line
          x1={courtWidth / 2}
          y1="0"
          x2={courtWidth / 2}
          y2={courtHeight}
          stroke="#ffffff"
          strokeWidth="3"
        />

        {/* Service boxes */}
        {/* Left service boxes */}
        <rect
          x="0"
          y="0"
          width={serviceBoxWidth}
          height={serviceBoxHeight}
          fill="#86efac"
          stroke="#22c55e"
          strokeWidth="1"
        />
        <rect
          x="0"
          y={serviceBoxHeight}
          width={serviceBoxWidth}
          height={serviceBoxHeight}
          fill="#86efac"
          stroke="#22c55e"
          strokeWidth="1"
        />

        {/* Right service boxes */}
        <rect
          x={courtWidth - serviceBoxWidth}
          y="0"
          width={serviceBoxWidth}
          height={serviceBoxHeight}
          fill="#86efac"
          stroke="#22c55e"
          strokeWidth="1"
        />
        <rect
          x={courtWidth - serviceBoxWidth}
          y={serviceBoxHeight}
          width={serviceBoxWidth}
          height={serviceBoxHeight}
          fill="#86efac"
          stroke="#22c55e"
          strokeWidth="1"
        />

        {/* Center service line */}
        <line
          x1="0"
          y1={serviceBoxHeight}
          x2={courtWidth}
          y2={serviceBoxHeight}
          stroke="#ffffff"
          strokeWidth="1"
        />

        {/* Service box dividers */}
        <line
          x1={serviceBoxWidth}
          y1="0"
          x2={serviceBoxWidth}
          y2={courtHeight}
          stroke="#ffffff"
          strokeWidth="1"
        />
        <line
          x1={courtWidth - serviceBoxWidth}
          y1="0"
          x2={courtWidth - serviceBoxWidth}
          y2={courtHeight}
          stroke="#ffffff"
          strokeWidth="1"
        />

        {/* Render shots */}
        {shots.map((shot) => (
          <ShotLine
            key={shot.id}
            shot={shot}
            courtWidth={courtWidth}
            courtHeight={courtHeight}
            onClick={() => onShotClick?.(shot)}
          />
        ))}
      </svg>
    </div>
  )
}
