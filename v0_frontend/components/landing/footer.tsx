import Image from "next/image"
import Link from "next/link"

const footerLinks = {
  product: [
    { name: "Features", href: "#features" },
    { name: "Pricing", href: "#pricing" },
    { name: "Demo", href: "#demo" },
  ],
  company: [
    { name: "About", href: "#about" },
    { name: "Blog", href: "#blog" },
    { name: "Careers", href: "#careers" },
  ],
  support: [
    { name: "Help Center", href: "#help" },
    { name: "Contact", href: "#contact" },
    { name: "Privacy", href: "#privacy" },
  ],
}

export function Footer() {
  return (
    <footer className="bg-[#0a0a0a] border-t border-[#333333]">
      <div className="max-w-7xl mx-auto px-4 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {/* Brand */}
          <div>
            <Link href="/" className="flex items-center gap-3 mb-4">
              <Image src="/logo.svg" alt="Courtvision Logo" width={40} height={40} className="w-10 h-10" />
              <span className="text-xl font-bold text-white">Courtvision</span>
            </Link>
            <p className="text-gray-400 text-sm">AI-Powered Tennis Analytics</p>
          </div>

          {/* Product */}
          <div>
            <h3 className="font-semibold text-white mb-4">Product</h3>
            <ul className="space-y-2">
              {footerLinks.product.map((link) => (
                <li key={link.name}>
                  <Link href={link.href} className="text-gray-400 hover:text-[#50C878] text-sm transition-colors">
                    {link.name}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Company */}
          <div>
            <h3 className="font-semibold text-white mb-4">Company</h3>
            <ul className="space-y-2">
              {footerLinks.company.map((link) => (
                <li key={link.name}>
                  <Link href={link.href} className="text-gray-400 hover:text-[#50C878] text-sm transition-colors">
                    {link.name}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Support */}
          <div>
            <h3 className="font-semibold text-white mb-4">Support</h3>
            <ul className="space-y-2">
              {footerLinks.support.map((link) => (
                <li key={link.name}>
                  <Link href={link.href} className="text-gray-400 hover:text-[#50C878] text-sm transition-colors">
                    {link.name}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="border-t border-[#333333] mt-12 pt-6">
          <p className="text-center text-gray-500 text-sm">
            &copy; {new Date().getFullYear()} Courtvision. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  )
}
