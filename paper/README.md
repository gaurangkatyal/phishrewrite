# Paper

LaTeX source for the PhishRewrite paper. This is a **working draft** scaffolded
from `RESULTS.md` and `LIMITATIONS.md`; every number in it traces to a CSV under
`results/tables/`.

## Build

```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Produces `main.pdf`. Needs a standard TeX distribution (TeX Live / MacTeX) with
`natbib`, `booktabs`, `siunitx`, `hyperref` — all in the default install.

## Layout

- `main.tex` — the paper (abstract → intro → related work → threat model/attack →
  setup → results → mitigation → limitations → ethics → conclusion).
- `references.bib` — bibliography.
- `figures/degradation_curve.png` — copied from `results/figures/`; regenerate the
  source with `python -m src.evaluate` if the numbers change.

## Before submitting (arXiv / venue)

- [ ] Verify dataset citations in `references.bib` — the `nazario`, `spamassassin`,
      and `champa2024ceas` entries carry repo provenance; confirm exact authors/years
      against the upstream sources.
- [ ] Decide on the general citation DOI (concept vs. version) for `phishrewrite2026`.
- [ ] Pick a venue style if not arXiv-only (the draft uses plain `article`).
- [ ] Add an author affiliation / ORCID and acknowledgements.
- [ ] Re-read the transformer §: the corpus-dependence framing (Nazario resists,
      CEAS-2008 degrades) is the load-bearing nuance — keep it honest.

## Numbers provenance

All headline figures come from the committed result tables (see `RESULTS.md` §9 for
the artifact index). The draft deliberately reports recall@0.5, detection@1%FPR,
and PR-AUC (not ROC-AUC) because ROC-AUC is insensitive under the class imbalance.
