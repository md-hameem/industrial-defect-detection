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
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ type: "spring", stiffness: 300, damping: 30 }}
      className="sticky top-0 z-50 px-4 pt-4"
    >
      <div
        className={`max-w-7xl mx-auto rounded-2xl border transition-colors duration-300 ${
          darkMode
            ? "bg-slate-900/40 backdrop-blur-2xl border-white/[0.06] shadow-[0_8px_32px_rgba(0,0,0,0.4)]"
            : "bg-white/70 backdrop-blur-2xl border-slate-200/60 shadow-lg shadow-slate-200/30"
        }`}
      >
        <div className="px-6 py-3.5 flex items-center justify-between">
          {/* Logo */}
          <Link href="/">
            <motion.div className="flex items-center gap-3 cursor-pointer" whileHover={{ scale: 1.02 }}>
              <div className="relative">
                {/* Glow halo */}
                <div className={`absolute inset-0 rounded-xl blur-xl ${darkMode ? "bg-blue-500/30" : "bg-blue-400/20"} glow-halo`} />
                <div className="relative w-10 h-10 bg-gradient-to-br from-blue-500 via-indigo-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg">
                  <Microscope className="w-5 h-5 text-white" />
                </div>
              </div>
              <div>
                <h1 className="text-lg font-bold text-gradient-animated">
                  DefectAI
                </h1>
                <p className={`text-[10px] tracking-wider uppercase ${darkMode ? "text-slate-500" : "text-slate-400"}`}>
                  Industrial Detection
                </p>
              </div>
            </motion.div>
          </Link>

          {/* Navigation */}
          <nav className="hidden md:flex items-center gap-0.5">
            {navItems.map((item) => {
              const isActive = pathname === item.href;
              const isHovered = hoveredItem === item.href;

              return (
                <Link key={item.href} href={item.href}>
                  <motion.div
                    onHoverStart={() => setHoveredItem(item.href)}
                    onHoverEnd={() => setHoveredItem(null)}
                    className={`relative px-4 py-2 rounded-xl font-medium text-sm transition-colors cursor-pointer ${
                      isActive
                        ? "text-white"
                        : darkMode
                        ? "text-slate-400 hover:text-slate-200"
                        : "text-slate-500 hover:text-slate-800"
                    }`}
                  >
                    {/* Active / Hover indicator */}
                    {(isActive || isHovered) && (
                      <motion.div
                        layoutId="navbar-indicator"
                        className={`absolute inset-0 rounded-xl ${
                          isActive
                            ? "bg-gradient-to-r from-blue-500/80 to-purple-600/80 shadow-lg shadow-blue-500/20"
                            : darkMode
                            ? "bg-white/[0.06]"
                            : "bg-slate-100"
                        }`}
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
              );
            })}
          </nav>

          {/* Theme Toggle */}
          <motion.button
            onClick={() => setDarkMode(!darkMode)}
            whileHover={{ scale: 1.1, rotate: 15 }}
            whileTap={{ scale: 0.9 }}
            className={`w-9 h-9 rounded-xl flex items-center justify-center transition-all cursor-pointer ${
              darkMode
                ? "bg-white/[0.06] hover:bg-white/[0.1] text-slate-400 hover:text-yellow-400"
                : "bg-slate-100 hover:bg-slate-200 text-slate-600 hover:text-indigo-600"
            }`}
          >
            {darkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </motion.button>
        </div>
      </div>

      {/* Mobile Navigation — Bottom Dock */}
      <div
        className={`md:hidden fixed bottom-4 left-4 right-4 z-50 rounded-2xl px-2 py-2 flex justify-around ${
          darkMode
            ? "bg-slate-900/60 backdrop-blur-2xl border border-white/[0.06] shadow-[0_-4px_24px_rgba(0,0,0,0.4)]"
            : "bg-white/80 backdrop-blur-2xl border border-slate-200/60 shadow-lg"
        }`}
      >
        {navItems.map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link key={item.href} href={item.href}>
              <motion.div
                whileTap={{ scale: 0.9 }}
                className={`relative px-3 py-2 rounded-xl text-center cursor-pointer ${
                  isActive
                    ? darkMode
                      ? "text-blue-400"
                      : "text-blue-600"
                    : darkMode
                    ? "text-slate-500"
                    : "text-slate-400"
                }`}
              >
                {isActive && (
                  <motion.div
                    layoutId="mobile-indicator"
                    className="absolute inset-0 bg-blue-500/10 rounded-xl"
                    transition={{ type: "spring", stiffness: 500, damping: 30 }}
                  />
                )}
                <item.icon className="w-5 h-5 mx-auto relative z-10" />
                <div className="text-[10px] mt-1 relative z-10 font-medium">{item.label}</div>
              </motion.div>
            </Link>
          );
        })}
      </div>
    </motion.header>
  );
}
