import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  devIndicators: false,
  output: "standalone",
  logging: {
    fetches: {
      fullUrl: false,
    },
  },
};

export default nextConfig;
