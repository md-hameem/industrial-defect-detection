"use client";

import { useState, useEffect } from "react";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

export default function ClientLayout({ children }: { children: React.ReactNode }) {
  const [darkMode, setDarkMode] = useState(true);

  useEffect(() => {
    const saved = localStorage.getItem("darkMode");
    if (saved !== null) setDarkMode(saved === "true");
  }, []);

  useEffect(() => {
    localStorage.setItem("darkMode", String(darkMode));
  }, [darkMode]);

  return (
    <div
      className={`min-h-screen flex flex-col transition-all duration-500 ${
        darkMode
          ? "bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950 text-white"
          : "bg-gradient-to-br from-slate-50 via-white to-blue-50 text-slate-900"
      }`}
    >
      {/* Animated Background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div
          className={`absolute -top-40 -right-40 w-80 h-80 rounded-full blur-3xl opacity-20 animate-pulse ${
            darkMode ? "bg-blue-500" : "bg-blue-300"
          }`}
        />
        <div
          className={`absolute -bottom-40 -left-40 w-80 h-80 rounded-full blur-3xl opacity-20 animate-pulse delay-1000 ${
            darkMode ? "bg-purple-500" : "bg-purple-300"
          }`}
        />
      </div>

      <Navbar darkMode={darkMode} setDarkMode={setDarkMode} />
      <main className="flex-1 relative">{children}</main>
      <Footer darkMode={darkMode} />
    </div>
  );
}
