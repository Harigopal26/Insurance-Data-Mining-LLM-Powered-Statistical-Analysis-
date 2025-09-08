
# 📊 Insurance Data Mining & LLM-Powered Statistical Analysis

This project leverages **Univariate**, **Bivariate**, and **Multivariate Analysis** techniques integrated with **LangChain**, **OpenAI**, and **RAG (Retrieval-Augmented Generation)** to uncover insights from an insurance dataset. It automates statistical computations, correlation analysis, regression modeling, and principal component analysis (PCA), enhanced by querying domain-specific PDFs using LLMs.

## 🔍 Project Modules

### 1. `UnivariateAnalysis.ipynb`
- Computes detailed summary statistics (mean, median, std, skewness, kurtosis, IQR, etc.)
- Normality checks with significance testing
- Group-wise comparison using t-tests / ANOVA
- RAG-powered insights using `analysisRag.pdf`

### 2. `Bivariate.ipynb`
- Correlation analysis (Pearson & Spearman)
- Chi-square and Cramer’s V for categorical associations
- Regression modeling (slope, intercept, R², RMSE)
- Group-level interaction assessments
- LLM-driven explanations for relationships

### 3. `Multivariate.ipynb`
- Correlation & partial correlation matrix across variables
- Multivariate normality checks using Mardia’s test
- Outlier detection via Mahalanobis distance (EllipticEnvelope)
- PCA for dimensionality reduction and interpretation
- Group-based comparative multivariate analysis
- RAG-based domain insights for variable interactions

## 📁 File Structure

```
.
├── UnivariateAnalysis.ipynb
├── Bivariate.ipynb
├── Multivariate.ipynb
├── data/
│   ├── insurance.csv                  # Main dataset
│   ├── datainfo.pdf                   # Dataset schema & metadata
│   └── analysisRag.pdf               # Domain knowledge base for RAG
├── integrated_analysis_report.txt    # Generated analysis report
├── README.md                         # Project documentation (this file)
```

## 💡 Key Features

- 🔢 Automated statistical pipelines with interpretability
- 🧠 LLM integration using LangChain agents
- 📚 RAG-powered contextual analysis from PDFs
- 📊 Visual and numeric outputs with fallback logic
- 📈 PCA for pattern extraction and variable reduction
- 📝 Report generation with conclusions and insights

## 🛠️ Requirements

Install dependencies via `poetry` or pip:

```bash
pip install pandas numpy scipy seaborn matplotlib scikit-learn statsmodels langchain openai faiss-cpu
```

Set your OpenAI key:

```bash
export OPENAI_API_KEY="your_api_key"
```

## 🚀 How to Run

Each notebook (`UnivariateAnalysis.ipynb`, `Bivariate.ipynb`, `Multivariate.ipynb`) is self-contained and can be executed individually.

For full automation, execute from command line:

```bash
python UnivariateAnalysis.ipynb
python Bivariate.ipynb
python Multivariate.ipynb
```

Ensure these files are present:
- `insurance.csv`
- `analysisRag.pdf`
- API key in environment

## 📈 Output

- Detailed printed outputs for each statistical layer
- Domain explanations via LLMs
- Final integrated report: `integrated_analysis_report.txt`

## 📘 References

- [LangChain Documentation](https://docs.langchain.com/)
- [OpenAI API](https://platform.openai.com/)
- [Pandas, NumPy, SciPy, Scikit-learn, Statsmodels]
