import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The 3D experience now lives at "/"; old /3d links keep working.
  async redirects() {
    return [{ source: "/3d", destination: "/", permanent: false }];
  },
};

export default nextConfig;
