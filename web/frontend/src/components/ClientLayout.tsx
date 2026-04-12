"use client";

import { useTheme } from "@/context/ThemeContext";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

export default function ClientLayout({ children }: { children: React.ReactNode }) {
  const { darkMode, setDarkMode } = useTheme();

  return (
    <div
      className={`min-h-screen flex flex-col transition-colors duration-500 relative ${
        darkMode
          ? "bg-[#020617] text-white"
          : "bg-gradient-to-br from-slate-50 via-white to-blue-50 text-slate-900"
      }`}
    >
      {/* Animated Mesh Gradient Background */}
      {darkMode && (
        <div className="fixed inset-0 overflow-hidden pointer-events-none">
          {/* Blob 1 - Blue */}
          <div
            className="mesh-blob-1 absolute w-[600px] h-[600px] rounded-full opacity-20"
            style={{
              background: "radial-gradient(circle, rgba(59,130,246,0.4) 0%, transparent 70%)",
              top: "-10%",
              left: "-5%",
              filter: "blur(80px)",
            }}
          />
          {/* Blob 2 - Purple */}
          <div
            className="mesh-blob-2 absolute w-[500px] h-[500px] rounded-full opacity-15"
            style={{
              background: "radial-gradient(circle, rgba(139,92,246,0.4) 0%, transparent 70%)",
              top: "40%",
              right: "-10%",
              filter: "blur(80px)",
            }}
          />
          {/* Blob 3 - Cyan */}
          <div
            className="mesh-blob-3 absolute w-[400px] h-[400px] rounded-full opacity-10"
            style={{
              background: "radial-gradient(circle, rgba(6,182,212,0.3) 0%, transparent 70%)",
              bottom: "0%",
              left: "30%",
              filter: "blur(80px)",
            }}
          />

          {/* Subtle Grid Overlay */}
          <div className="absolute inset-0 grid-overlay opacity-50" />

          {/* Vignette */}
          <div
            className="absolute inset-0"
            style={{
              background: "radial-gradient(ellipse at center, transparent 50%, rgba(2,6,23,0.8) 100%)",
            }}
          />
        </div>
      )}

      {/* Light mode subtle background */}
      {!darkMode && (
        <div className="fixed inset-0 overflow-hidden pointer-events-none">
          <div
            className="mesh-blob-1 absolute w-[500px] h-[500px] rounded-full opacity-30"
            style={{
              background: "radial-gradient(circle, rgba(59,130,246,0.15) 0%, transparent 70%)",
              top: "-10%",
              right: "10%",
              filter: "blur(60px)",
            }}
          />
          <div
            className="mesh-blob-2 absolute w-[400px] h-[400px] rounded-full opacity-20"
            style={{
              background: "radial-gradient(circle, rgba(139,92,246,0.1) 0%, transparent 70%)",
              bottom: "10%",
              left: "-5%",
              filter: "blur(60px)",
            }}
          />
        </div>
      )}

      <Navbar darkMode={darkMode} setDarkMode={setDarkMode} />
      <main className="flex-1 relative z-10">{children}</main>
      <Footer darkMode={darkMode} />
    </div>
  );
}
