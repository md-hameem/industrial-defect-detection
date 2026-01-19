"use client";

import { createContext, useContext, useState, useEffect, ReactNode } from "react";

interface ThemeContextType {
  darkMode: boolean;
  setDarkMode: (value: boolean) => void;
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [darkMode, setDarkMode] = useState(true);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    const saved = localStorage.getItem("darkMode");
    if (saved !== null) setDarkMode(saved === "true");
    setMounted(true);
  }, []);

  useEffect(() => {
    if (mounted) {
      localStorage.setItem("darkMode", String(darkMode));
    }
  }, [darkMode, mounted]);

  const toggleTheme = () => setDarkMode(!darkMode);

  // Prevent flash of wrong theme
  if (!mounted) {
    return <div className="min-h-screen bg-slate-950" />;
  }

  return (
    <ThemeContext.Provider value={{ darkMode, setDarkMode, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error("useTheme must be used within a ThemeProvider");
  }
  return context;
}

// Theme-aware class helper
export function cn(darkClass: string, lightClass: string, darkMode: boolean): string {
  return darkMode ? darkClass : lightClass;
}
