"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";
import { Home, Search, History, BarChart3, Info, Sun, Moon, Microscope } from "lucide-react";

interface NavbarProps {
  darkMode: boolean;
  setDarkMode: (value: boolean) => void;
}

const navItems = [
  { href: "/", label: "Home", icon: Home },
  { href: "/detect", label: "Detect", icon: Search },
  { href: "/history", label: "History", icon: History },
  { href: "/research", label: "Research", icon: BarChart3 },
  { href: "/about", label: "About", icon: Info },
];

export default function Navbar({ darkMode, setDarkMode }: NavbarProps) {
  const pathname = usePathname();
  const [hoveredItem, setHoveredItem] = useState<string | null>(null);

  return (
    <motion.header
      initial={{ y: -50, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      className={`border-b backdrop-blur-xl sticky top-0 z-50 ${
        darkMode ? "border-white/10 bg-slate-900/80" : "border-slate-200/60 bg-white/80"
      }`}
    >
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        {/* Logo */}
        <Link href="/">
          <motion.div className="flex items-center gap-4 cursor-pointer" whileHover={{ scale: 1.02 }}>
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl blur-lg opacity-50" />
              <div className="relative w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg">
                <Microscope className="w-6 h-6 text-white" />
              </div>
            </div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                DefectAI
              </h1>
              <p className={`text-xs ${darkMode ? "text-slate-400" : "text-slate-600"}`}>
                Industrial Detection
              </p>
            </div>
          </motion.div>
        </Link>

        {/* Navigation */}
        <nav className="hidden md:flex items-center gap-1">
          {navItems.map((item) => (
            <Link key={item.href} href={item.href}>
              <motion.div
                onHoverStart={() => setHoveredItem(item.href)}
                onHoverEnd={() => setHoveredItem(null)}
                className={`relative px-4 py-2 rounded-xl font-medium transition-colors ${
                  pathname === item.href
                    ? "text-white"
                    : darkMode
                    ? "text-slate-400 hover:text-white"
                    : "text-slate-600 hover:text-slate-900"
                }`}
              >
                {(pathname === item.href || hoveredItem === item.href) && (
                  <motion.div
                    layoutId="navbar-indicator"
                    className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl"
                    initial={false}
                    transition={{ type: "spring", stiffness: 500, damping: 30 }}
                  />
                )}
                <span className="relative z-10 flex items-center gap-2">
                  <item.icon className="w-4 h-4" />
                  <span>{item.label}</span>
                </span>
              </motion.div>
            </Link>
          ))}
        </nav>

        {/* Theme Toggle */}
        <motion.button
          onClick={() => setDarkMode(!darkMode)}
          whileHover={{ scale: 1.1, rotate: 15 }}
          whileTap={{ scale: 0.9 }}
          className={`w-10 h-10 rounded-xl flex items-center justify-center transition-colors ${
            darkMode ? "bg-slate-800 hover:bg-slate-700" : "bg-slate-100 hover:bg-slate-200"
          }`}
        >
          {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
        </motion.button>
      </div>

      {/* Mobile Navigation */}
      <div className="md:hidden border-t border-white/5 px-4 py-2 flex justify-around">
        {navItems.map((item) => (
          <Link key={item.href} href={item.href}>
            <motion.div
              whileTap={{ scale: 0.95 }}
              className={`px-3 py-2 rounded-lg text-center ${
                pathname === item.href
                  ? "bg-blue-500/20 text-blue-400"
                  : darkMode
                  ? "text-slate-400"
                  : "text-slate-600"
              }`}
            >
              <item.icon className="w-5 h-5 mx-auto" />
              <div className="text-xs mt-1">{item.label}</div>
            </motion.div>
          </Link>
        ))}
      </div>
    </motion.header>
  );
}
