import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // A self-contained server in .next/standalone: what web/Dockerfile ships.
  output: "standalone",
};

export default nextConfig;
