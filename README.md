# Fluorosis – VDR Gene Variability, Fluoride Exposure & Fluorosis Study

**COMSATS University Islamabad | MS Biochemistry and Molecular Biology**

> *Association of VDR Gene Variability, Fluoride Exposure and Fluorosis Among Children
> Population of Thal Desert Areas*
>
> **Student:** Andleeb Zahra · **Reg. No.:** FA24-RBM-003
> **Supervisor:** Dr. Ismat Nawaz · **Co-supervisor:** Dr. Syed Ali Musstjab Akber Shah Eqani

---

## Repository Structure

```
fluorosis/
├── COMSATS-MS-Format-Thesis 3.tex  ← LaTeX source for the MS Synopsis/Thesis
├── COMSATS-MS-Format-Thesis 3.pdf  ← Compiled PDF (10 pages)
├── README.md                        ← This file
├── app.py                           ← Streamlit web application (risk calculator)
├── fluorosis_calculator.py          ← Data generation, statistical analysis & ML model
├── fluorosis_model.pkl              ← Trained XGBoost model
├── data_new.xlsx                    ← Dataset (4 sheets)
├── analysis_graphs.png              ← Publication-quality figures (300 DPI)
├── shap_result.png                  ← SHAP feature-importance plot (300 DPI)
└── requirements.txt                 ← Python dependencies
```

---

## Thesis PDF — Status & How to Generate

### Current status

Both the LaTeX source and the compiled PDF are tracked in the repository at the **root level**:

| File | Description |
|------|-------------|
| `COMSATS-MS-Format-Thesis 3.tex` | LaTeX source (edit this to update the thesis) |
| `COMSATS-MS-Format-Thesis 3.pdf` | Compiled output — 10 pages, ready to read or print |

If you modify the `.tex` source and need to recompile, follow one of the options below.

---

### Option 1 – Overleaf (recommended, no local install needed)

1. Go to [https://www.overleaf.com](https://www.overleaf.com) and sign in (free account is sufficient).
2. Click **New Project → Upload Project**.
3. Upload `COMSATS-MS-Format-Thesis 3.tex` (zip it first if Overleaf asks for a zip).
4. Overleaf compiles automatically. Click **Recompile** if needed.
5. Click the **Download PDF** button (top-right) to save `COMSATS-MS-Format-Thesis 3.pdf`.

---

### Option 2 – pdflatex (TeX Live or MikTeX, any OS)

Requires a local TeX distribution (see installation links below).

```bash
# The filename contains a space, so quote it:
pdflatex "COMSATS-MS-Format-Thesis 3.tex"
# Run twice to resolve cross-references:
pdflatex "COMSATS-MS-Format-Thesis 3.tex"
```

Output: **`COMSATS-MS-Format-Thesis 3.pdf`** in the same directory.

Auxiliary files created during compilation (safe to delete afterwards):

| File | Purpose |
|------|---------|
| `COMSATS-MS-Format-Thesis 3.aux` | Cross-reference data |
| `COMSATS-MS-Format-Thesis 3.log` | Compilation log |
| `COMSATS-MS-Format-Thesis 3.out` | Hyperref bookmarks |

---

### Option 3 – TeX Live (Linux / macOS)

**Install TeX Live:**

```bash
# Ubuntu / Debian
sudo apt-get install texlive-full

# macOS (via Homebrew)
brew install --cask mactex
```

Then compile as shown in Option 2.

---

### Option 4 – MikTeX (Windows)

1. Download and install MikTeX from [https://miktex.org/download](https://miktex.org/download).
2. Open **MikTeX Console** → allow automatic package installation.
3. Open a command prompt in the repository folder and run:

```cmd
pdflatex "COMSATS-MS-Format-Thesis 3.tex"
pdflatex "COMSATS-MS-Format-Thesis 3.tex"
```

4. The PDF **`COMSATS-MS-Format-Thesis 3.pdf`** will appear in the same folder.

---

## Python Web Application

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the risk calculator

```bash
streamlit run app.py
```

### Regenerate dataset, graphs, and ML model

```bash
python fluorosis_calculator.py
```

---

## Citation

If you use this code or data in your research, please cite:

> Zahra, A. (2025). *Association of VDR Gene Variability, Fluoride Exposure and Fluorosis
> Among Children Population of Thal Desert Areas* [MS Synopsis].
> COMSATS University Islamabad, Department of Biosciences.
