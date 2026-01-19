import { Skeleton } from "@/components/LoadingUI";

export default function Loading() {
  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      {/* Header skeleton */}
      <div className="mb-12">
        <div className="flex items-center gap-3 mb-4">
          <Skeleton className="w-10 h-10 rounded" />
          <Skeleton className="h-10 w-64" />
        </div>
        <Skeleton className="h-6 w-96" />
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="p-6 rounded-2xl border border-white/10 bg-slate-800/50 text-center">
            <Skeleton className="w-8 h-8 mx-auto mb-2" />
            <Skeleton className="h-10 w-16 mx-auto mb-2" />
            <Skeleton className="h-4 w-24 mx-auto" />
          </div>
        ))}
      </div>

      {/* Figures gallery */}
      <div className="mb-12">
        <Skeleton className="h-8 w-40 mb-6" />
        <div className="grid md:grid-cols-2 gap-6">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="rounded-2xl border border-white/10 bg-slate-800/50 overflow-hidden">
              <Skeleton className="h-48 w-full" />
              <div className="p-4">
                <Skeleton className="h-5 w-32 mb-2" />
                <Skeleton className="h-3 w-48" />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Table skeleton */}
      <Skeleton className="h-8 w-48 mb-6" />
      <div className="rounded-2xl border border-white/10 overflow-hidden">
        <div className="bg-slate-800/80 p-4">
          <div className="grid grid-cols-3 gap-4">
            <Skeleton className="h-4 w-24" />
            <Skeleton className="h-4 w-20" />
            <Skeleton className="h-4 w-32" />
          </div>
        </div>
        {[1, 2, 3, 4, 5].map((i) => (
          <div key={i} className="p-4 border-t border-white/5">
            <div className="grid grid-cols-3 gap-4">
              <Skeleton className="h-4 w-24" />
              <Skeleton className="h-6 w-16 rounded-lg" />
              <Skeleton className="h-3 w-full rounded-full" />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
