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
      className={`border-t mt-auto ${darkMode ? "border-slate-800" : "border-slate-200"}`}
    >
      <div className="max-w-7xl mx-auto px-6 py-6">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          <p className={darkMode ? "text-slate-500" : "text-slate-400"}>
            © 2026 Industrial Defect Detection • Bachelor&apos;s Thesis | Made by Mohammad Hamim - 202280090114 
          </p>
          <div className="flex gap-4">
            <a
              href="https://github.com/md-hameem/industrial-defect-detection"
              target="_blank"
              rel="noopener noreferrer"
              className={`hover:text-blue-400 transition-colors ${
                darkMode ? "text-slate-500" : "text-slate-400"
              }`}
            >
              GitHub
            </a>
            <span className={darkMode ? "text-slate-700" : "text-slate-300"}>•</span>
            <span className={darkMode ? "text-slate-500" : "text-slate-400"}>
              Zhengzhou University
            </span>
          </div>
        </div>
      </div>
    </motion.footer>
  );
}
