/**
 * Compile-time upper bound on fused chain length.
 *
 * Lives in its own module so the shader emitter (fusion-jit.ts) and the
 * circuit (quantum.ts) can both depend on it without introducing a cycle.
 * Raising this costs 32 B per slot in the uniform buffer — cheap, bounded
 * by maxUniformBufferBindingSize (typically 64 KiB).
 */
export const FUSED_CHAIN_MAX_K = 64;
