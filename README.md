# bronchial_cytology_analysis
This study identified lung cancer risk factors in C3 and C4 levels and developed a scoring model to identify high-risk patients.

## Data Structure
The pipeline expects input data in Excel format (.xlsx). The data should include patient records with lab results, flags, and outcomes. Below is the required column structure:

- **Patient number**: Unique identifier (string or int).
- **Sex**: Gender ('M' or 'F').
- **Date of Birth (yyyy-mm-dd)**: Birth date in YYYY-MM-DD format.
- **Admission Date (yyyy-mm-dd)**: Admission date in YYYY-MM-DD format.
- **Lab Result Columns** (floats, may include NaN): e.g., APTT_Result, Albumin_Result, Basophil, absolute_Result, C-Reactive Protein_Result, etc.
- **Flagging Columns** (strings: 'H', 'L', or blank/NaN): e.g., APTT_Flagging, Albumin_Flagging, etc. (indicating high/low/normal).
- **Lung cancer**: Target outcome (int: 0 or 1).

**Sample Row** (from data.xlsx):
| Patient number | Sex | Date of Birth (yyyy-mm-dd) | Admission Date (yyyy-mm-dd) | APTT_Result | ... | Lung cancer |
|----------------|-----|----------------------------|-----------------------------|-------------|-----|-------------|
| 123           | F   | 1950-12-19                | 2023-03-30                 | 27.1       | ... | 0          |
