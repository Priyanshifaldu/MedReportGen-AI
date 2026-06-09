# MedReportGen-AI

AI-powered clinical report generation system that transforms structured patient data into medically coherent draft reports using fine-tuned transformer models, explainability techniques, and automated validation mechanisms.

---

## Overview

Clinical documentation is a time-consuming process that often increases administrative workload for healthcare professionals. MedReportGen-AI addresses this challenge by automatically generating structured medical reports from patient information while maintaining transparency, explainability, and human oversight.

The system leverages Natural Language Processing (NLP), Transformer-based language models, and Explainable AI techniques to generate clinically plausible reports from structured healthcare data.

---

## Key Features

### Automated Report Generation
- Converts structured patient data into natural language medical reports
- Generates clinically formatted summaries
- Produces draft reports ready for physician review

### Explainable AI
- SHAP-based feature attribution
- Confidence scoring
- Identification of key contributing clinical factors

### Clinical Validation
- Factual consistency checking
- Medical language validation
- Confidence estimation for generated reports

### User Interface
- Interactive dashboard
- Patient information input forms
- Report generation and review interface
- Explainability visualizations

### Report Management
- Save generated reports
- Export reports
- Review report history
- Compare outputs

---

## System Architecture

```text
Structured Patient Data
            │
            ▼
Data Cleaning & Anonymization
            │
            ▼
Structured-to-Text Conversion
            │
            ▼
Fine-Tuned Transformer Model
            │
            ▼
Medical Validation Layer
            │
            ▼
Clinician Review Interface
            │
            ▼
Final Medical Report
```

---

## Technology Stack

### Frontend

- React
- Vite
- JavaScript
- HTML/CSS

### Backend

- Python
- Flask/FastAPI
- Transformers
- Hugging Face
- SHAP

### Machine Learning

- ClinicalT5
- T5
- Transformer Models
- Explainable AI

### Deployment

- Docker
- GitHub

---

## Project Structure

```text
MedReportGen-AI
│
├── frontend
│   ├── src
│   ├── public
│   ├── assets
│   ├── package.json
│   └── vite.config.js
│
├── backend
│   ├── app
│   ├── data
│   ├── model
│   ├── train.py
│   ├── evaluate_model.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── evaluation_results_t5_gpu.json
├── final_evaluation_results.json
└── README.md
```

---

## Workflow

### Step 1
Input structured patient information:

- Demographics
- Vital signs
- Laboratory values
- Clinical notes
- Diagnostic codes

### Step 2
Data preprocessing:

- Cleaning
- Standardization
- Normalization
- Medical code conversion

### Step 3
Model inference:

- Clinical language model generates report
- Context-aware medical reasoning
- Structured report creation

### Step 4
Validation:

- Consistency checks
- Confidence scoring
- Explainability analysis

### Step 5
Physician review and approval

---

## Explainability Features

The system provides:

- Feature importance scores
- SHAP visualizations
- Confidence estimates
- Key contributing factors
- Transparent decision support

---

## Sample Output Sections

Generated reports may contain:

- Patient Summary
- Clinical Findings
- Laboratory Results
- Diagnostic Interpretation
- Treatment Recommendations
- Follow-up Suggestions

---

## Evaluation Metrics

The project evaluates generated reports using:

### NLP Metrics

- BLEU Score
- ROUGE-L
- BERTScore

### Clinical Metrics

- Contextual Accuracy
- Human Readability
- Clinical Consistency
- Factual Validation

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/MedReportGen-AI.git
cd MedReportGen-AI
```

---

### Backend Setup

```bash
cd backend

pip install -r requirements.txt
```

Run server:

```bash
python app.py
```

---

### Frontend Setup

```bash
cd frontend

npm install
npm run dev
```

---

## Research Foundation

This project is inspired by research in:

- ClinicalT5
- MIMIC-CXR
- RadGraph
- Medical NLP
- Transformer-based Clinical Text Generation
- Explainable AI in Healthcare

---

## Future Enhancements

- Multi-modal report generation
- Medical image integration
- Real-time clinical assistance
- Hospital EHR integration
- Advanced fact-checking
- Cross-hospital deployment
- Improved clinical explainability

---

## Project Outcomes

- Automated clinical documentation
- Reduced physician documentation burden
- Improved report consistency
- Enhanced workflow efficiency
- Explainable AI-assisted healthcare reporting

---

## Authors

### Priyanshi Faldu
B.Tech Artificial Intelligence & Machine Learning

### Shreyans Gadekar

### Lakshya Joshi

---

## Disclaimer

This project is intended for research and educational purposes only.

Generated reports should always be reviewed and approved by qualified healthcare professionals before clinical use.

---

## License

MIT License
