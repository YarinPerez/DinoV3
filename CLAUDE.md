# Claude Implementation Guidelines

## Overview
This document outlines the required process for implementing programs in this project. The primary goal is **teaching and research**, not just task completion. All implementations must be extensively documented to help readers understand and learn.

---

## Pre-Implementation Requirements

### 1. Product Requirements Document (PRD)
**Location:** `docs/PRD.md`

**Requirements:**
- Must be created BEFORE any implementation begins
- Must be approved by the user before proceeding
- Should be extensive and educational, helping readers understand the program thoroughly

**Content Guidelines:**

#### For Machine Learning Models:
- Confusion matrix plots
- Performance analysis (accuracy, precision, recall, F1-score, etc.)
- Results must be saved to files for later inspection
- Include visualizations in the README
- Explain model architecture and choices

#### For Statistical Work (e.g., Clustering):
- Include graphs and visualizations in README
- Report standard deviations and statistical measures
- Provide detailed information about clusters
- Use dimensionality reduction algorithms (PCA, t-SNE, UMAP) when necessary for visualization
- Explain statistical methodology

#### For LLM Usage:
- Calculate and report token usage
- Estimate costs
- Document prompt engineering strategies
- Show example inputs/outputs

### 2. Implementation Tasks Document
**Location:** `docs/TASKS.md`

**Requirements:**
- Created AFTER PRD approval
- Must be approved before implementation begins
- Lists all implementation tasks in detail
- Tasks should be marked as done during implementation

### 3. Planning and Architecture Document
**Location:** `docs/PLANNING.md`

**Requirements:**
- Created AFTER PRD approval
- Must be approved before implementation begins
- Details implementation planning and architecture
- Describes how tasks will be tracked (e.g., marking done tasks in TASKS.md)
- Includes system architecture diagrams/descriptions
- Explains design decisions and trade-offs

---

## Implementation Requirements

### Approval Process
**CRITICAL:** Do not implement ANY code until:
1. PRD is created and approved
2. TASKS.md is created and approved
3. PLANNING.md is created and approved

### Python Environment
- All Python code must run under `uv` virtual environment
- Code must be compiled to improve runtime performance
- Use appropriate compilation tools (e.g., Cython, Numba, or similar)

### Code Organization
- **Maximum 150 lines per source code file**
- Split larger modules into smaller, focused files
- Maintain clear separation of concerns
- Use descriptive file and function names

### Documentation Requirements
- All results must be saved to files for later inspection
- Include visualizations and analysis in README
- Code must be well-commented for educational purposes
- Provide clear examples and use cases

---

## Workflow Summary

```
1. Create PRD (docs/PRD.md)
   ↓
2. User approves PRD
   ↓
3. Create TASKS.md (docs/TASKS.md)
   ↓
4. Create PLANNING.md (docs/PLANNING.md)
   ↓
5. User approves TASKS.md and PLANNING.md
   ↓
6. Begin implementation
   ↓
7. Mark tasks as done in TASKS.md
   ↓
8. Save results and generate visualizations
   ↓
9. Update README with findings
```

---

## Remember
The goal is to **teach and help readers understand**, not just to complete tasks. Every implementation should serve as a learning resource.
