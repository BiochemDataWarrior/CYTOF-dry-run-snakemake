## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# CyTOF Data Analysis Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![Snakemake](https://img.shields.io/badge/snakemake-≥7.0-brightgreen)](https://snakemake.github.io/)

A scalable end-to-end computational pipeline for processing, normalizing, clustering, and performing differential abundance testing on mass cytometry (CyTOF) FCS datasets.

<Image src="image_agent_tag_14366003773629253647" alt="CyTOF data processing and downstream analysis flowchart" caption="Mass cytometry data processing and analysis workflow" />

---

## Features

- **Raw FCS Parsing & Preprocessing:** Quality control filtering, bead-based normalization, and debarcoding.
- **Data Transformation:** Generalized arcsinh transformation ($cofactor = 5$) with customizable per-channel parameters.
- **Batch Correction & Harmonization:** CytoNorm / Harmony integration to adjust for run-to-run variation.
- **Unsupervised Clustering:** High-speed cell population identification using FlowSOM and PhenoGraph algorithms.
- **Dimensionality Reduction:** Automated UMAP / t-SNE projection generation.
- **Differential Analysis:** Differential abundance (DA) and differential state (DS) testing via GLMM / diffcyt frameworks.

---

## Pipeline Overview
---

## Installation

### Prerequisites

Ensure you have Conda / Mamba installed.

```bash
# Clone the repository
git clone [https://github.com/your-username/cytof-pipeline.git](https://github.com/your-username/cytof-pipeline.git)
cd cytof-pipeline

# Create and activate environment
conda env create -f environment.yml
conda activate cytof-env

# config/panel.csv example
fcs_colname,antigen,marker_class
141Pr,CD19,type
143Nd,CD45RA,state
144Nd,CD11b,type

snakemake --cores 8

snakemake --profile profiles/slurm



├── config/
│   ├── config.yaml          # Main configuration file
│   ├── metadata.csv         # Sample metadata (condition, batch, subject)
│   └── panel.csv            # Antibody panel definition
├── workflow/
│   ├── Snakefile            # Main pipeline workflow definition
│   └── scripts/             # Python & R transformation/clustering modules
├── results/                 # Output FCS files, RDS objects, and heatmaps
└── LICENSE                  # MIT License



---

### Step-by-Step Instructions to Set This Up on GitHub

<Sequence>
  <Step title="Add the LICENSE file">
    In your GitHub repository, click **Add file** > **Create new file**. Type `LICENSE` into the filename box, click **Choose a license template**, select **MIT License**, fill in your copyright year and name, and click **Commit changes...**.
  </Step>
  <Step title="Add or update README.md">
    Create or edit your `README.md` file, paste the template above, and adjust the repository URLs, configuration paths, or dependencies to match your pipeline's environment (`R`, `Python`, `Snakemake`, etc.).
  </Step>
  <Step title="Commit changes">
    Scroll to the bottom of the page and click **Commit changes...** to publish your updated documentation.
  </Step>
</Sequence>

<FollowUp label="Would you like an example environment.yml file or Snakemake rule set to include alongside t
