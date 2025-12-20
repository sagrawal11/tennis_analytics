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
    <section className="py-24 bg-black">
      <div className="max-w-7xl mx-auto px-4">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-white mb-4">Trusted by Coaches and Players</h2>
          <p className="text-xl text-gray-400">See what our users are saying</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {testimonials.map((testimonial) => (
            <div key={testimonial.author} className="bg-[#1a1a1a] rounded-xl p-8 border border-[#333333]">
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
