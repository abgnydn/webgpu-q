/** Thin wrapper that localizes the single `(navigator as unknown as { gpu?: GPU })` cast. */
export function getWebGPU(): { gpu?: GPU } {
  return navigator as unknown as { gpu?: GPU };
}
