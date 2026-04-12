"use client";

import { motion } from "framer-motion";

interface FooterProps {
  darkMode: boolean;
}

export default function Footer({ darkMode }: FooterProps) {
  return (
    <motion.footer
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className={`relative z-10 mt-auto ${
        darkMode ? "border-t border-white/[0.04]" : "border-t border-slate-200/60"
      }`}
    >
      {/* Top glow line */}
      {darkMode && (
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-1/2 h-px bg-gradient-to-r from-transparent via-blue-500/30 to-transparent" />
      )}

      <div className="max-w-7xl mx-auto px-6 py-6">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          <p className={`text-sm ${darkMode ? "text-slate-600" : "text-slate-400"}`}>
            © 2026 Industrial Defect Detection • Bachelor&apos;s Thesis | Made by Mohammad Hamim - 202280090114
          </p>
          <div className="flex gap-4 items-center">
            <a
              href="https://github.com/md-hameem/industrial-defect-detection"
              target="_blank"
              rel="noopener noreferrer"
              className={`text-sm transition-colors cursor-pointer ${
                darkMode
                  ? "text-slate-600 hover:text-blue-400"
                  : "text-slate-400 hover:text-blue-600"
              }`}
            >
              GitHub
            </a>
            <span className={darkMode ? "text-slate-800" : "text-slate-300"}>•</span>
            <span className={`text-sm ${darkMode ? "text-slate-600" : "text-slate-400"}`}>
              Zhengzhou University
            </span>
          </div>
        </div>
      </div>
    </motion.footer>
  );
}
