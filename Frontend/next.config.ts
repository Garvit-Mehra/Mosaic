import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  devIndicators: false,
  output: "standalone", // Required for Docker deployment
};

export default nextConfig;
