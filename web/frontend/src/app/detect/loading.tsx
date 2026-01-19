import { Skeleton } from "@/components/LoadingUI";

export default function Loading() {
  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      {/* Header skeleton */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <Skeleton className="w-12 h-12 rounded-xl" />
          <Skeleton className="h-10 w-64" />
        </div>
        <Skeleton className="h-4 w-96" />
      </div>

      {/* Two column layout skeleton */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
        {/* Left panel */}
        <div className="lg:col-span-2 space-y-6">
          <div className="p-6 rounded-2xl border border-white/10 bg-slate-800/50">
            <Skeleton className="h-5 w-40 mb-4" />
            <div className="grid grid-cols-2 gap-3">
              <Skeleton className="h-24 rounded-xl" />
              <Skeleton className="h-24 rounded-xl" />
            </div>
          </div>
          <div className="p-6 rounded-2xl border border-white/10 bg-slate-800/50">
            <Skeleton className="h-5 w-32 mb-4" />
            <div className="grid grid-cols-3 gap-2 mb-4">
              <Skeleton className="h-16 rounded-xl" />
              <Skeleton className="h-16 rounded-xl" />
              <Skeleton className="h-16 rounded-xl" />
            </div>
            <Skeleton className="h-12 w-full rounded-xl" />
          </div>
        </div>

        {/* Right panel */}
        <div className="lg:col-span-3">
          <div className="p-6 rounded-2xl border border-white/10 bg-slate-800/50 min-h-[600px]">
            <Skeleton className="h-6 w-48 mb-6" />
            <div className="flex items-center justify-center h-96">
              <div className="text-center">
                <Skeleton className="w-16 h-16 rounded-full mx-auto mb-4" />
                <Skeleton className="h-4 w-32 mx-auto mb-2" />
                <Skeleton className="h-3 w-48 mx-auto" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
