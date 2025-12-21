const testimonials = [
  {
    quote: "Courtvision has transformed how we analyze our players' performance. The insights are incredible.",
    author: "Sarah Johnson",
    role: "Head Coach, Stanford Tennis",
  },
  {
    quote: "As a player, seeing my shot patterns visualized helps me understand my game so much better.",
    author: "Michael Chen",
    role: "Professional Player",
  },
  {
    quote: "The team management features make it easy to track all my players in one place.",
    author: "David Martinez",
    role: "College Tennis Director",
  },
]

export function TestimonialsSection() {
  return (
    <section className="py-24 relative overflow-hidden">
      {/* Base background */}
      <div className="absolute inset-0 bg-black" />
      {/* Subtle green glow throughout - stronger in middle, fades at edges */}
      <div className="absolute inset-0 bg-gradient-to-br from-[#50C878]/6 via-[#50C878]/8 to-[#50C878]/6 pointer-events-none" />
      {/* Decorative glow accents */}
      <div className="absolute top-1/3 right-1/3 w-[500px] h-[500px] bg-[#50C878]/5 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-1/3 left-1/3 w-[500px] h-[500px] bg-[#50C878]/5 rounded-full blur-3xl pointer-events-none" />
      {/* Fade green glow out at top to preserve black border */}
      <div className="absolute top-0 left-0 right-0 h-[200px] bg-gradient-to-b from-black to-transparent pointer-events-none" />
      {/* Fade to pure black at bottom (200px) - for footer transition */}
      <div className="absolute bottom-0 left-0 right-0 h-[200px] bg-gradient-to-b from-transparent to-black pointer-events-none" />
      <div className="max-w-7xl mx-auto px-4 relative z-10">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-white mb-4">Trusted by Coaches and Players</h2>
          <p className="text-xl text-gray-400">See what our users are saying</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {testimonials.map((testimonial) => (
            <div key={testimonial.author} className="bg-[#1a1a1a] rounded-xl p-8 border border-[#333333] transition-all duration-200 ease-in-out hover:border-[#50C878]/50 hover:shadow-lg hover:shadow-[#50C878]/10 transform hover:scale-[1.02]">
              <p className="text-lg text-white italic mb-6">&quot;{testimonial.quote}&quot;</p>
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-full bg-[#50C878]/20 flex items-center justify-center">
                  <span className="text-[#50C878] font-semibold text-lg">{testimonial.author[0]}</span>
                </div>
                <div>
                  <p className="font-semibold text-white">{testimonial.author}</p>
                  <p className="text-sm text-gray-400">{testimonial.role}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
