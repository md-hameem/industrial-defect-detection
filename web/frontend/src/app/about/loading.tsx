import { Skeleton } from "@/components/LoadingUI";

export default function Loading() {
  return (
    <div className="max-w-6xl mx-auto px-6 py-8">
      {/* Header */}
      <div className="text-center mb-16">
        <Skeleton className="h-12 w-64 mx-auto mb-4" />
        <Skeleton className="h-6 w-96 mx-auto" />
      </div>

      {/* Project Overview */}
      <div className="p-8 rounded-3xl border border-white/10 bg-slate-800/30 mb-16">
        <div className="flex items-center justify-center gap-3 mb-6">
          <Skeleton className="w-12 h-12 rounded-xl" />
          <Skeleton className="h-8 w-40" />
        </div>
        <Skeleton className="h-4 w-full max-w-4xl mx-auto mb-2" />
        <Skeleton className="h-4 w-3/4 max-w-4xl mx-auto" />
      </div>

      {/* Features Grid */}
      <div className="mb-16">
        <Skeleton className="h-8 w-40 mx-auto mb-8" />
        <div className="grid md:grid-cols-3 gap-6">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <div key={i} className="p-6 rounded-2xl border border-white/5 bg-slate-800/30">
              <Skeleton className="w-12 h-12 rounded-xl mb-4" />
              <Skeleton className="h-5 w-32 mb-2" />
              <Skeleton className="h-3 w-full mb-1" />
              <Skeleton className="h-3 w-2/3" />
            </div>
          ))}
        </div>
      </div>

      {/* Author section */}
      <div className="p-6 rounded-2xl border border-white/10 bg-slate-800/30">
        <div className="grid md:grid-cols-2 gap-6">
          {[1, 2].map((i) => (
            <div key={i} className="flex items-center gap-4">
              <Skeleton className="w-16 h-16 rounded-full shrink-0" />
              <div>
                <Skeleton className="h-3 w-16 mb-1" />
                <Skeleton className="h-6 w-40 mb-1" />
                <Skeleton className="h-3 w-32" />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
