"use client";

import { useEffect } from "react";

import { Button } from "@/components/ui";

// A page that throws while rendering shows this instead of a blank screen.
export default function PageError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => console.error(error), [error]);
  return (
    <div className="space-y-3 rounded-lg border border-red-200 bg-red-50 p-4 dark:border-red-900 dark:bg-red-950">
      <h2 className="font-semibold text-red-900 dark:text-red-200">
        This page hit an error
      </h2>
      <p className="font-mono text-sm text-red-800 dark:text-red-300">{error.message}</p>
      <Button variant="secondary" onClick={reset}>
        Try again
      </Button>
    </div>
  );
}
