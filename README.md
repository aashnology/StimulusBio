# StimulusBio (WIP)

### An Open-Source Computational Framework for Analysing Cellular Responses to External Stimuli

StimulusBio is an open-source framework for building **reproducible, validated, and interpretable computational workflows** for studying cellular responses to controlled external stimuli.

The initial implementation focuses on **bulk transcriptomic data from stimulus-response experiments**.

> **External stimulus → Cellular response → Molecular measurements → Computational analysis → Biological interpretation**

## Why StimulusBio?

Biological data analysis can produce misleading results when problems in the underlying data or experimental design go undetected.

StimulusBio therefore follows a **validation-first** approach, with emphasis on:

* Experimental-design awareness
* Data integrity and sample alignment
* Reproducibility
* Appropriate statistical methodology
* Effect sizes and uncertainty
* Transparent provenance
* Conservative biological interpretation

The framework is designed to **fail loudly rather than silently produce questionable analyses**.

## Current Status

**Early development — V0.1**

The current implementation provides a validation layer for expression data and experimental metadata.

### Implemented

* Expression-matrix structural validation
* Duplicate feature detection
* Duplicate sample detection
* Missing-value detection
* Non-numeric value detection
* Metadata schema validation
* Duplicate metadata sample detection
* Missing condition-label detection
* Expression/metadata sample-alignment checks
* Feature-level filtering by explicit mean-expression threshold
* Expression normalization (CPM, log2, per-feature z-score)
* Per-sample QC diagnostics (library size, missingness, zero-count)
* Exploratory PCA (principal-component scores per sample, explained variance)
* Automated tests for validation, preprocessing, and exploratory-analysis behaviour

**Current test status: 107 tests passing, 3 skipped (pending Steps 7/9/10)**

### Planned V0.1 Workflow

```text
Data ingestion
      ↓
Validation
      ↓
Preprocessing
      ↓
Quality control
      ↓
Exploratory analysis
      ↓
Differential expression
      ↓
Pathway / gene-set analysis
      ↓
Visualisation
      ↓
Reproducible reporting
```

## Input

The initial framework is designed for datasets containing:

**Expression matrix**

* Stable feature/gene identifiers
* Sample identifiers
* Numeric expression measurements

**Experimental metadata**

* `sample_id`
* `condition`

Additional metadata such as `batch`, `cell_type`, `timepoint`, and `treatment` can be supported where available.

## Scientific Principles

1. **Validate before analysing**
2. **Reproducibility over convenience**
3. **Effect sizes and uncertainty alongside significance**
4. **Experimental design matters**
5. **No unsupported causal claims**
6. **Negative or inconclusive results are valid**
7. **Analysis provenance should be traceable**

## Case Study

StimulusBio is a **framework, not a single dataset analysis**.

A public stimulus-response transcriptomic dataset will be used as an initial case study to demonstrate the framework end-to-end, including:

* Dataset provenance
* Experimental design
* Validation
* Quality control
* Statistical analysis
* Biological interpretation
* Limitations
* Reproducibility

## Repository Structure

```text
StimulusBio/
├── src/stimulusbio/
│   ├── io/
│   ├── validation/
│   ├── preprocessing/
│   ├── statistics/
│   ├── visualization/
│   └── reporting/
├── tests/
├── notebooks/
├── examples/
├── data/
├── docs/
│   ├── methodology/
│   └── research/
├── README.md
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml
└── .gitignore
```

## Development Philosophy

StimulusBio is being developed incrementally.

Each component should have a clear purpose, defined inputs and outputs, automated tests, documented assumptions, and scientific justification.

Infrastructure will be added **when required by the research workflow**, rather than built speculatively.

## Future Directions

Potential extensions include:

* Time-series stimulus-response analysis
* Additional omics modalities
* Single-cell transcriptomics
* Cross-dataset comparison
* Predictive modelling
* Response classification
* Automated analytical reporting

These will be considered after the core workflow is established and validated.

## Contributing

Contributions, scientific feedback, methodological discussion, and reproducibility improvements are welcome.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for contribution guidelines.

## License

MIT License.
