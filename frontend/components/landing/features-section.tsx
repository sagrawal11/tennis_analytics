import { Target, BarChart3, Users, Zap, LayoutGrid, Video } from "lucide-react"

const features = [
  {
    icon: Target,
    title: "Precise Shot Tracking",
    description: "Every shot mapped with computer vision powered accuracy for detailed analysis.",
  },
  {
    icon: BarChart3,
    title: "Performance Analytics",
    description: "Deep insights into your game patterns and improvement areas.",
  },
  {
    icon: Users,
    title: "Team Collaboration",
    description: "Coaches and players working together seamlessly.",
  },
  {
    icon: Zap,
    title: "Instant Analysis",
    description: "Get insights in minutes, not hours. Real-time processing.",
  },
  {
    icon: LayoutGrid,
    title: "Interactive Court View",
    description: "See your shots visualized on an interactive court diagram.",
  },
  {
    icon: Video,
    title: "Video Playback",
    description: "Watch key moments synced with your shot data.",
  },
]

export function FeaturesSection() {
  return (
    <section className="py-24 relative overflow-hidden">
      {/* Base background */}
      <div className="absolute inset-0 bg-[#0a0a0a]" />
      {/* Subtle green glow throughout - stronger in middle, fades at edges */}
      <div className="absolute inset-0 bg-gradient-to-br from-[#50C878]/8 via-[#50C878]/6 to-[#50C878]/4 pointer-events-none" />
      {/* Decorative glow accents */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-[#50C878]/5 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-[#50C878]/5 rounded-full blur-3xl pointer-events-none" />
      {/* Fade green glow out at top to preserve black border */}
      <div className="absolute top-0 left-0 right-0 h-[200px] bg-gradient-to-b from-black to-transparent pointer-events-none" />
      {/* Fade green glow out at bottom to preserve black border */}
      <div className="absolute bottom-0 left-0 right-0 h-[200px] bg-gradient-to-b from-transparent to-black pointer-events-none" />
      <div className="max-w-7xl mx-auto px-4 relative z-10">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-white mb-4">Everything You Need to Improve</h2>
          <p className="text-xl text-gray-400">Powerful analytics tools for coaches and players</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature) => (
            <div
              key={feature.title}
              className="bg-[#1a1a1a] rounded-xl p-8 border border-[#333333] hover:border-[#50C878]/50 transition-all duration-200 ease-in-out group hover:shadow-lg hover:shadow-[#50C878]/10 transform hover:scale-[1.02]"
            >
              <feature.icon className="h-10 w-10 text-[#50C878] mb-4 group-hover:scale-110 transition-transform" />
              <h3 className="text-xl font-semibold text-white mb-2">{feature.title}</h3>
              <p className="text-gray-400">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
