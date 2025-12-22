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
    <section className="py-24 relative overflow-visible">
      {/* Base background */}
      <div className="absolute inset-0 bg-[#0a0a0a]" />
      {/* Subtle green glow throughout - reduced intensity */}
      <div className="absolute inset-0 bg-gradient-to-br from-[#50C878]/4 via-[#50C878]/3 to-[#50C878]/2 pointer-events-none" />
      {/* Bridging glow elements from hero section (extending from top) - reduced intensity */}
      <div className="absolute -top-[200px] left-1/4 w-[600px] h-[400px] bg-[#50C878]/4 rounded-full blur-3xl pointer-events-none z-20" />
      <div className="absolute -top-[150px] right-1/3 w-[500px] h-[350px] bg-[#50C878]/3 rounded-full blur-3xl pointer-events-none z-20" />
      {/* Decorative glow accents in middle - reduced intensity */}
      <div className="absolute top-1/3 left-1/3 w-[400px] h-[400px] bg-[#50C878]/3 rounded-full blur-3xl pointer-events-none z-10" />
      <div className="absolute bottom-1/3 right-1/3 w-[450px] h-[450px] bg-[#50C878]/3 rounded-full blur-3xl pointer-events-none z-10" />
      {/* Bridging glow elements to testimonials (extending to bottom) - reduced intensity */}
      <div className="absolute bottom-0 left-1/5 w-[550px] h-[400px] bg-[#50C878]/4 rounded-full blur-3xl pointer-events-none z-20" />
      <div className="absolute bottom-0 right-1/4 w-[600px] h-[380px] bg-[#50C878]/3 rounded-full blur-3xl pointer-events-none z-20" />
      {/* Fade green glow out at top to preserve black border - on top */}
      <div className="absolute top-0 left-0 right-0 h-[200px] bg-gradient-to-b from-black to-transparent pointer-events-none z-30" />
      {/* Fade green glow out at bottom to preserve black border - on top */}
      <div className="absolute bottom-0 left-0 right-0 h-[200px] bg-gradient-to-b from-transparent to-black pointer-events-none z-30" />
      <div className="max-w-7xl mx-auto px-4 relative z-40">
        <div className="text-center mb-16 relative">
          {/* Pure black background behind heading to ensure white text */}
          <div className="absolute inset-0 bg-black/50 -mx-4 -my-2 blur-sm pointer-events-none" />
          <h2 className="text-4xl font-bold mb-4 relative z-10" style={{ color: '#ffffff', WebkitTextFillColor: '#ffffff' }}>Everything You Need to Improve</h2>
          <p className="text-xl text-gray-400 relative z-10">Powerful analytics tools for coaches and players</p>
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
