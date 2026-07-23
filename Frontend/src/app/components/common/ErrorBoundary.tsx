"use client";

import React from "react";

interface Props {
  children: React.ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

export class ErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error("ErrorBoundary caught:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center h-full gap-4 p-8">
          <div className="text-[var(--color2)] text-xl font-semibold">
            Something went wrong
          </div>
          <p className="text-[var(--color3)] text-sm text-center max-w-md">
            An unexpected error occurred. Try refreshing the page.
          </p>
          {this.state.error && (
            <pre className="text-xs text-red-400 bg-[var(--input-bg)] p-3 rounded-xl max-w-lg overflow-auto">
              {this.state.error.message}
            </pre>
          )}
          <button
            onClick={() => {
              this.setState({ hasError: false, error: undefined });
              window.location.reload();
            }}
            className="px-4 py-2 rounded-xl bg-[var(--color2)] text-[var(--color4)] text-sm font-medium hover:opacity-80 transition-opacity cursor-pointer"
          >
            Reload Page
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
