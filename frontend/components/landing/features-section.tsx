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
    <section className="py-24 bg-[#0a0a0a]">
      <div className="max-w-7xl mx-auto px-4">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-white mb-4">Everything You Need to Improve</h2>
          <p className="text-xl text-gray-400">Powerful analytics tools for coaches and players</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature) => (
            <div
              key={feature.title}
              className="bg-[#1a1a1a] rounded-xl p-8 border border-[#333333] hover:border-[#50C878]/50 transition-colors group"
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
