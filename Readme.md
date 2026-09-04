# Coverage-aware H* adsorption on Pt(111) and the HER volcano with universal ML interatomic potentials

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/isocan/Pt111-MACE/blob/main/Pt111_H_coverage_differential_UMA_v5.ipynb)

Reproducible, notebook-based workflows for adsorption energetics on Pt(111) using
universal machine-learned interatomic potentials (MLIPs) as RPBE-level screening
models. The repository accompanies the **HİDROKAP** proposal
(*Machine Learning-Assisted Discovery of Hydrogen Coverage- and Solvent-Aware
Bimetallic HER Electrocatalysts*) and serves as its preliminary study.

The central question: **how much does the HER descriptor ΔG<sub>H*</sub> change
when hydrogen coverage and lateral interactions are included, and can a universal
MLIP reproduce the reference RPBE-DFT energetics well enough to screen with?**

---

## Main notebook — `Pt111_H_coverage_differential_UMA_v5.ipynb`

An AdsorbML-style configuration funnel for H* on a four-layer **4×4 Pt(111)** slab
(64 Pt, 16 surface sites) with **UMA-S-1.1** (`task_name="oc20"`, RPBE-trained), followed by a
Nørskov-style 12-metal benchmark and HER volcano analysis. Runs end-to-end on a free Colab
GPU in roughly one hour.

### What it does

1. **RPBE-consistent geometry.** The Pt lattice constant is derived from the OC20 RPBE-relaxed
   bulk record (`mp-126`, a = 3.990 Å); no PBE/RPBE mixing.
2. **Four hydrogen coverages:** 1H, 4H, 8H, 16H → θ<sub>H</sub> = 1/16, 1/4, 1/2, 1 ML.
3. **Configuration sampling** per coverage from three streams — enumerated high-symmetry
   sites, greedy periodic *maximin* separation, and FairChem `random_site_heuristic_placement`
   / `random` — 313 candidates in total.
4. **Funnel:** UMA single-point prescreen → strategy-balanced relaxation (BFGS, 0.05 eV/Å) →
   anomaly detection (dissociation, desorption, surface change, intercalation) → **geometric
   site classification** (ontop / bridge / fcc / hcp, decided from first-, second- and
   third-layer neighbours, independent of any site catalogue).
5. **Energetics:** cumulative, average and coverage-step (block differential) ΔE<sub>H</sub>
   referenced to ½ H<sub>2</sub>; coverage-specific harmonic ZPE and entropy → ΔG<sub>H*</sub>
   at 298 K, 1 bar.
6. **Literature validation** against Nørskov *et al.* (2005) at both coverage-matched points
   (0.25 ML and 1 ML).
7. **12-metal Nørskov benchmark:** for Au, Ag, Pd, Pt, Rh, Ir, Ni, W, Co, Cu, Mo, Re the
   RPBE bulk is pulled from the OC20 database by Materials Project ID, the Nørskov geometry
   is rebuilt (2×2, three layers, 1 H = 0.25 ML, close-packed plane chosen from the detected
   lattice), every high-symmetry site is relaxed, and UMA is compared with the Nørskov DFT,
   EquiformerV2 and single-point DFT values; parity, per-metal and **HER volcano** figures.
8. **Hand-off:** VASP (`GGA = RP`) inputs for clean, H<sub>2</sub> and every covered slab, a
   template for RPBE energies, and a state-level template for solvent corrections
   (OC25/eSEN, explicit or implicit solvation) — MLIP and DFT energies are never mixed in one cycle.

### Key results (UMA/OC20-RPBE)

| θ<sub>H</sub> (ML) | Minimum | Average ΔE<sub>H</sub> (eV/H) | Step ΔE<sub>H</sub> (eV/H) | Step ΔG<sub>H*</sub> (eV/H) | Nørskov 2005 avg. ΔE<sub>H</sub> |
|---|---|---|---|---|---|
| 1/16 | 1×fcc | −0.339 | −0.339 | −0.149 | — |
| 1/4 | 4×fcc | −0.326 | −0.322 | −0.138 | −0.33 |
| 1/2 | 8×fcc | −0.291 | −0.255 | −0.061 | — |
| 1 | 16×fcc | −0.238 | −0.185 | +0.011 | −0.27 |

* The fcc three-fold hollow is the minimum at every coverage; the H–H repulsion weakens
  the step energy by ≈0.15 eV from 1/16 to 1 ML, and ΔG<sub>H*</sub> crosses zero only at 1 ML.
* Coverage-matched agreement with Nørskov *et al.* RPBE: 0.004 eV (0.25 ML) and 0.032 eV (1 ML).
* 12-metal set, mean absolute error vs. Nørskov DFT: **UMA (Nørskov site) 0.028 eV**,
  EquiformerV2 0.042 eV, single-point DFT 0.063 eV (max. UMA error 0.08 eV).

### Figures

All figures are written to `Pt111_H_4x4_UMA_RPBE_v5/figures/` as 600-dpi PNG + PDF + SVG
(English and Turkish label sets):

| File | Content |
|---|---|
| `S1_Pt111_kaplama_egilimi` | Average and step ΔE<sub>H</sub> vs. coverage, Nørskov points overlaid |
| `S2_Pt111_kaplama_eslestirilmis_parite` | Coverage-matched ΔE / ΔG parity bars |
| `S3_Norskov_seti_parite` | 12-metal parity plot (UMA, EquiformerV2, SP-DFT vs. Nørskov) |
| `S4_Norskov_seti_metal_karsilastirma` | Per-metal grouped bars |
| `S5_HER_volkan_Norskov_seti` | HER volcano: experimental log i<sub>0</sub> vs. ΔE<sub>H</sub> descriptor |
| `S6_Norskov_seti_yapilar_ustten` | Top-view gallery of the 12 UMA minima |
| `selected_Pt111_structures_static` | Top/side views of the 0–1 ML Pt(111) minima with classified sites |
| `thermochemistry_decomposition`, `configuration_screening_funnel` | ΔE/ZPE/−TΔS breakdown; sampling-stream statistics |

### Outputs

Every stage is exported as CSV under `Pt111_H_4x4_UMA_RPBE_v5/` (generated, prescreened,
selected, accepted and rejected candidates; strategy minima; electronic and free-energy
summaries; 12-metal benchmark and error tables) together with all ASE trajectories, so the
funnel is fully auditable.

---

## Supporting notebooks

| Notebook | Purpose |
|---|---|
| `Pt111_O_OH_OOH_MACE_Colab.ipynb` | O*, OH* and OOH* adsorption on Pt(111) with MACE — ORR intermediates, single-site demonstration |
| `FairChem_UMA_FineTuning_Tutorial_Colab.ipynb` | Tutorial on fine-tuning UMA with FairChem for a custom DFT dataset (used as the starting point for the proposal's fine-tuning work package) |

---

## Running

Open the main notebook in Colab (badge above), select a **GPU runtime**, and run all cells.
The first cell installs `fairchem-core`, `fairchem-data-oc`, `ase`, `pandas`, `matplotlib`
and `pymatgen`; the UMA checkpoint (~1.2 GB) is downloaded from Hugging Face on first use
(a token is only needed if the download returns 401). Main switches at the top of the notebook:

```python
COVERAGE_STATES_NH = (1, 4, 8, 16)   # coverages to sample
RELAX_QUOTAS       = {...}           # relaxations per sampling stream — increase for production
RUN_VIBRATIONS     = True            # ZPE / entropy (16H adds ~100 single points)
RUN_NORSKOV_SET    = True            # 12-metal benchmark
RUN_RPBE_POSTPROCESS, RUN_SOLVENT_POSTPROCESS = False, False   # enable after filling the templates
```

Local execution works the same way with a CUDA-capable GPU; set `DEVICE="cpu"` for testing only.

## Method notes and caveats

* UMA is used strictly as an RPBE-level *screening* model; quantitative claims require the
  full RPBE(-D3) DFT cycle exported by the notebook.
* Coverage-step energies for 1→4, 4→8 and 8→16 H are per-H block differentials between
  sampled minima, not true n-th-H differentials.
* The `oc20` task has no explicit spin input; Ni/Co magnetism is implicit in the model.
* OC20 records `mp-102` (Co) and `mp-8642` (Re) are fcc, so "Co(111)" and "Re(111)" are fcc(111) surfaces, consistent with Nørskov's labels.
* OC20 energies are solid–gas; solvent effects enter only through the explicit state-level template.

## References

* J. K. Nørskov, T. Bligaard, A. Logadottir, J. R. Kitchin, J. G. Chen, S. Pandelov, U. Stimming, *J. Electrochem. Soc.* **152**, J23 (2005). [doi:10.1149/1.1856988](https://doi.org/10.1149/1.1856988)
* B. M. Wood *et al.*, UMA: A Family of Universal Models for Atoms, 2025. [arXiv:2506.23971](https://arxiv.org/abs/2506.23971)
* L. Chanussot *et al.*, Open Catalyst 2020 (OC20) Dataset and Community Challenges, *ACS Catal.* **11**, 6059 (2021).
* J. Lan *et al.*, AdsorbML: a leap in efficiency for adsorption energy calculations using generalizable machine learning potentials, *npj Comput. Mater.* **9**, 172 (2023).
* I. Batatia *et al.*, MACE: Higher order equivariant message passing neural networks, *NeurIPS* 2022.


## License

MIT (see `LICENSE`).
