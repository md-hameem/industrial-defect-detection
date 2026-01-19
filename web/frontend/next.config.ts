import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactCompiler: true,
  
  // Image optimization
  images: {
    // Enable optimized base64 encoded images from API
    unoptimized: true, // Since we use base64 encoded images from API
    
    // Remote patterns for external images (if needed)
    remotePatterns: [
      {
        protocol: 'http',
        hostname: 'localhost',
        port: '8000',
      },
    ],
    
    // Supported formats for optimization
    formats: ['image/webp', 'image/avif'],
  },
  
  // Performance optimizations
  experimental: {
    // Enable parallel routes
    parallelServerCompiles: true,
    parallelServerBuildTraces: true,
  },
  
  // Compress responses
  compress: true,
  
  // Production optimizations
  poweredByHeader: false,
};

export default nextConfig;
