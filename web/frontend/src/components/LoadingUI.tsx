"use client";

import { motion } from "framer-motion";
import { useTheme } from "@/context/ThemeContext";

// Skeleton loading component
export function Skeleton({ className = "" }: { className?: string }) {
  const { darkMode } = useTheme();
  return (
    <div className={`animate-pulse rounded ${darkMode ? "bg-slate-700/50" : "bg-slate-200"} ${className}`} />
  );
}

// Card skeleton
export function CardSkeleton() {
  const { darkMode } = useTheme();
  return (
    <div className={`p-6 rounded-2xl border ${darkMode ? "bg-slate-800/50 border-white/10" : "bg-white/80 border-slate-200"}`}>
      <Skeleton className="h-4 w-1/3 mb-4" />
      <Skeleton className="h-32 w-full mb-3" />
      <Skeleton className="h-3 w-full mb-2" />
      <Skeleton className="h-3 w-2/3" />
    </div>
  );
}

// Result skeleton (for detection results)
export function ResultSkeleton() {
  const { darkMode } = useTheme();
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`p-4 rounded-xl border ${darkMode ? "bg-slate-800/30 border-white/5" : "bg-slate-50 border-slate-200"}`}
    >
      <div className="flex items-center justify-between mb-4">
        <Skeleton className="h-6 w-20" />
        <Skeleton className="h-8 w-24" />
      </div>
      <div className="grid grid-cols-3 gap-3">
        <Skeleton className="aspect-square rounded-lg" />
        <Skeleton className="aspect-square rounded-lg" />
        <Skeleton className="aspect-square rounded-lg" />
      </div>
      <div className="mt-4 flex justify-between items-center">
        <Skeleton className="h-4 w-32" />
        <Skeleton className="h-8 w-20 rounded-lg" />
      </div>
    </motion.div>
  );
}

// Image upload skeleton
export function UploadSkeleton() {
  const { darkMode } = useTheme();
  return (
    <div className={`border-2 border-dashed rounded-2xl p-8 text-center ${darkMode ? "border-white/10 bg-slate-800/30" : "border-slate-300 bg-slate-50"}`}>
      <Skeleton className="w-16 h-16 rounded-full mx-auto mb-4" />
      <Skeleton className="h-4 w-48 mx-auto mb-2" />
      <Skeleton className="h-3 w-32 mx-auto" />
    </div>
  );
}

// Spinner for button loading states
export function Spinner({ size = "md" }: { size?: "sm" | "md" | "lg" }) {
  const sizeClass = size === "sm" ? "w-4 h-4" : size === "lg" ? "w-8 h-8" : "w-5 h-5";
  return (
    <svg className={`animate-spin ${sizeClass}`} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
    </svg>
  );
}

// Progress bar
export function ProgressBar({ current, total }: { current: number; total: number }) {
  const { darkMode } = useTheme();
  const percent = total > 0 ? (current / total) * 100 : 0;
  
  return (
    <div className="space-y-2">
      <div className="flex justify-between text-sm">
        <span className={darkMode ? "text-slate-400" : "text-slate-600"}>Processing...</span>
        <span className={darkMode ? "text-white" : "text-slate-900"}>{current}/{total}</span>
      </div>
      <div className={`h-2 rounded-full overflow-hidden ${darkMode ? "bg-slate-700" : "bg-slate-200"}`}>
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${percent}%` }}
          transition={{ duration: 0.3 }}
          className="h-full bg-gradient-to-r from-blue-500 to-purple-600"
        />
      </div>
    </div>
  );
}

// Error alert component
export function ErrorAlert({ message, onDismiss }: { message: string; onDismiss?: () => void }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 flex items-start gap-3"
    >
      <svg className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
      <div className="flex-1">
        <p className="text-rose-400 font-medium">Error</p>
        <p className="text-rose-300/80 text-sm">{message}</p>
      </div>
      {onDismiss && (
        <button onClick={onDismiss} className="text-rose-400 hover:text-rose-300 p-1">
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      )}
    </motion.div>
  );
}

// Success message
export function SuccessAlert({ message }: { message: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center gap-3"
    >
      <svg className="w-5 h-5 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
      <p className="text-emerald-400">{message}</p>
    </motion.div>
  );
}

// Retry button
export function RetryButton({ onClick, loading }: { onClick: () => void; loading?: boolean }) {
  return (
    <button
      onClick={onClick}
      disabled={loading}
      className="flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-500 hover:bg-blue-600 text-white disabled:opacity-50 transition"
    >
      {loading ? <Spinner size="sm" /> : (
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
        </svg>
      )}
      Retry
    </button>
  );
}
