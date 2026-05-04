# ─────────────────────────────────────────────────────────────
# itensor-reference.jl — generate ground-state energies and
# bond-dim profiles for the textbook 1-D models we ship in TS,
# using ITensor's reference DMRG implementation.
#
# The output JSON is committed to tests/manybody/itensor-reference.json
# and used as ground truth in tests/manybody/itensor-ref.test.ts.
#
# Run:    julia --project=. tools/itensor-reference.jl
# Or:     julia tools/itensor-reference.jl   (uses default global env)
# ─────────────────────────────────────────────────────────────

using ITensors, ITensorMPS, JSON3

function heisenberg_mpo(sites, J)
  N = length(sites)
  os = OpSum()
  # H = J Σ S_i · S_{i+1} = J Σ [S^z S^z + (1/2)(S^+ S^- + S^- S^+)]
  for j in 1:N-1
    os += J,        "Sz", j, "Sz", j+1
    os += 0.5 * J,  "S+", j, "S-", j+1
    os += 0.5 * J,  "S-", j, "S+", j+1
  end
  return MPO(os, sites)
end

function tfim_mpo(sites, J, h)
  # Pauli convention: ITensor's "S=1/2" sites give Sz = σz/2, Sx = σx/2.
  # Our TS Hamiltonian uses Pauli matrices directly: H_TS = -J Σ Z_iZ_{i+1} - h Σ X_i.
  # Translate: Z_i = 2 Sz_i, X_i = 2 Sx_i, so:
  #   H_TS = -4J Σ Sz_iSz_{i+1} - 2h Σ Sx_i
  N = length(sites)
  os = OpSum()
  for j in 1:N-1
    os += -4.0 * J, "Sz", j, "Sz", j+1
  end
  for j in 1:N
    os += -2.0 * h, "Sx", j
  end
  return MPO(os, sites)
end

function xxz_mpo(sites, J, delta)
  # H_TS = J Σ (X_iX_{i+1} + Y_iY_{i+1} + Δ Z_iZ_{i+1}).
  # In Sz-Sx-Sy: X = 2Sx, Y = 2Sy, Z = 2Sz, so XX = 4 SxSx, etc.
  N = length(sites)
  os = OpSum()
  for j in 1:N-1
    os += 4.0 * J,         "Sx", j, "Sx", j+1
    os += 4.0 * J,         "Sy", j, "Sy", j+1
    os += 4.0 * J * delta, "Sz", j, "Sz", j+1
  end
  return MPO(os, sites)
end

function run_dmrg(model_name, mpo_factory, model_args, N; chi_max=200, sweeps=10, etol=1e-9)
  sites = siteinds("S=1/2", N)
  H = mpo_factory(sites, model_args...)
  psi0 = random_mps(sites; linkdims=10)

  # ITensorMPS expects setmaxdim!(::Sweeps, Int...) — splat the schedule.
  sw = Sweeps(sweeps)
  setmaxdim!(sw, 10, 20, 40, 80, 200, chi_max)
  setcutoff!(sw, 1e-12)

  energy, psi = dmrg(H, psi0, sw; outputlevel=0)

  # Bond dimensions actually used (at the converged state).
  link_dims = [dim(linkind(psi, j)) for j in 1:N-1]

  return Dict(
    "model" => model_name,
    "args" => Dict(string(i) => v for (i, v) in enumerate(model_args)),
    "N" => N,
    "energy" => energy,
    "energy_per_site" => energy / N,
    "max_bond_dim" => isempty(link_dims) ? 0 : maximum(link_dims),
    "bond_dims" => link_dims,
    "dmrg_sweeps" => sweeps,
    "chi_cap" => chi_max,
  )
end

function main()
  results = Dict[]

  # Heisenberg chains: tiny + textbook.
  for N in [4, 8, 12, 20, 40]
    push!(results, run_dmrg("heisenberg", heisenberg_mpo, (1.0,), N))
    println("heisenberg N=$N done")
  end

  # TFIM, three slices: ferromagnetic (h<<J), critical (h=J), paramagnetic (h>>J).
  for (h, label) in [(0.5, "ferro"), (1.0, "critical"), (2.0, "para")]
    for N in [8, 16, 32]
      r = run_dmrg("tfim", tfim_mpo, (1.0, h), N)
      r["regime"] = label
      push!(results, r)
      println("tfim h=$h N=$N done")
    end
  end

  # XXZ at a couple of Δ.
  for delta in [0.5, 1.0, 1.5]
    push!(results, run_dmrg("xxz", xxz_mpo, (1.0, delta), 16))
    println("xxz Δ=$delta N=16 done")
  end

  # Bethe-ansatz limit reference (analytical, not from DMRG):
  # E_0/N → 1/4 - ln(2) ≈ -0.4431 as N → ∞ for Heisenberg.
  bethe = Dict(
    "model" => "heisenberg",
    "kind" => "bethe-ansatz-limit",
    "energy_per_site" => 0.25 - log(2),
    "N" => "infinite",
  )
  push!(results, bethe)

  # Pfeuty (1970) closed form for TFIM at h=J=1 (critical), thermodynamic limit:
  # E_0/N = -2/π. Open chain has finite-size corrections.
  pfeuty = Dict(
    "model" => "tfim",
    "kind" => "pfeuty-critical-limit",
    "energy_per_site" => -2 / π,
    "N" => "infinite",
    "args" => Dict("1" => 1.0, "2" => 1.0),
  )
  push!(results, pfeuty)

  out = Dict(
    "generated_by" => "tools/itensor-reference.jl",
    "itensors_version" => string(pkgversion(ITensors)),
    "results" => results,
  )

  open("tests/manybody/itensor-reference.json", "w") do io
    JSON3.pretty(io, out)
  end
  println("wrote tests/manybody/itensor-reference.json — $(length(results)) entries")
end

main()
