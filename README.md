# Portfolio Analytics RAG System

A Retrieval-Augmented Generation (RAG) system for querying financial portfolio data using semantic search and natural language processing. This system combines traditional data filtering with vector embeddings to answer complex questions about trades, holdings, and portfolio performance.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [System Architecture](#system-architecture)
4. [Setup Instructions](#setup-instructions)
5. [Configuration](#configuration)
6. [Usage](#usage)
7. [Test Results](#test-results)
8. [Technical Implementation](#technical-implementation)

---

## Overview

This system processes financial data from CSV files containing trade records and portfolio holdings. It enables users to query the data using natural language questions and receive accurate, formatted responses. The system uses a hybrid approach combining:

- Semantic search using sentence transformers and FAISS indexing
- Traditional dataframe filtering based on extracted entities
- LLM-based response formatting via OpenRouter API

**Key Capabilities:**
- Count operations (e.g., "How many trades for portfolio X?")
- Aggregations (e.g., "Total PL YTD for fund Y?")
- Rankings (e.g., "Which fund has the best quarterly profit?")
- Complex filtering across multiple dimensions
- Fuzzy matching for portfolio names and securities

---

## Repository Structure

```
.
├── data/
│   ├── holdings.csv          # Portfolio holdings data
│   ├── trades.csv             # Trade transaction data
│   └── cache/                 # Auto-generated cache directory (gitignored)
│       ├── chunks.pkl         # Cached text chunks
│       └── chunk_embeddings.pkl  # Cached embeddings
│
├── main.ipynb                 # Main Jupyter notebook with complete implementation
├── requirements.txt           # Python dependencies
├── .env.example              # Template for environment variables
├── .env                      # Your actual API key (gitignored)
└── README.md                 # This file
```

**Data Files:**
- `holdings.csv`: Contains portfolio positions with columns like PortfolioName, SecurityId, Qty, Price, PL_YTD, PL_MTD, PL_QTD, MV_Base
- `trades.csv`: Contains trade transactions with columns like PortfolioName, TradeTypeName, SecurityId, Quantity, Price, Principal

**Cache Directory:**
The `data/cache/` directory is automatically created when the system runs and stores:
- Preprocessed text chunks from portfolios and securities
- Vector embeddings for semantic search
- This caching significantly improves subsequent query performance

---

## System Architecture

### High-Level Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER NATURAL LANGUAGE QUERY                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    QUERY CLASSIFICATION                         │
│  - Dataset Detection (trades/holdings/both)                     │
│  - Operation Type (count/aggregate/rank/show)                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ENTITY EXTRACTION                            │
│  - Portfolio name (with fuzzy matching)                         │
│  - Ticker symbol                                                │
│  - Trade type (Buy/Sell)                                        │
│  - Metrics (PL_YTD, MV_Base, etc.)                              │
│  - Conditions (negative values, etc.)                           │
└────────────┬───────────────────────────────────┬────────────────┘
             │                                   │
             ▼                                   ▼
┌──────────────────────────┐      ┌──────────────────────────────┐
│  SEMANTIC SEARCH         │      │  DIRECT DATA FILTERING       │
│  - Embed query           │      │  - Filter by portfolio       │
│  - Search FAISS index    │      │  - Filter by ticker          │
│  - Retrieve top chunks   │      │  - Filter by trade type      │
│  - Score relevance       │      │  - Apply conditions          │
└──────────┬───────────────┘      └──────────┬───────────────────┘
           │                                 │
           │                                 ▼
           │                      ┌──────────────────────────────┐
           │                      │  OPERATION EXECUTION         │
           │                      │  - COUNT: Count filtered rows│
           │                      │  - AGGREGATE: Sum/Mean       │
           │                      │  - RANK: Sort & select top N │
           │                      └──────────┬───────────────────┘
           │                                 │
           └─────────────┬───────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RESPONSE GENERATION                          │
│  - Combine summary with evidence                                │
│  - Format via LLM (OpenRouter API)                              │
│  - Apply strict formatting rules                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FINAL FORMATTED ANSWER                       │
└─────────────────────────────────────────────────────────────────┘
```

### Processing Pipeline

1. **Data Loading & Preprocessing**
   - Load CSV files with type safety
   - Normalize text fields (lowercase, strip whitespace)
   - Convert numeric columns with error handling

2. **Chunk Generation**
   - Create portfolio summaries (aggregated metrics, top holdings, top trades)
   - Create security summaries (holdings and trades per security)
   - Generate global summary (top portfolios by various metrics)

3. **Embedding & Indexing**
   - Use sentence-transformers (all-MiniLM-L6-v2) to embed chunks
   - Build FAISS index with normalized vectors for cosine similarity
   - Cache embeddings for performance

4. **Query Processing**
   - Extract entities using pattern matching and fuzzy string matching
   - Classify query to determine dataset and operation type
   - Execute appropriate operation (count/aggregate/rank/show)

5. **Response Formatting**
   - Generate summary from operation results
   - Retrieve relevant evidence chunks via semantic search
   - Format response using LLM with strict rules to prevent hallucination

---

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Jupyter Notebook or JupyterLab
- OpenRouter API account (free tier available)

### Step-by-Step Installation

1. **Clone or download this repository**

2. **Install dependencies**

   **IMPORTANT:** Review `requirements.txt` before installation to ensure compatibility with your environment.

   ```bash
   pip install -r requirements.txt
   ```

   Required packages:
   - `pandas`: Data manipulation
   - `sentence-transformers`: Text embeddings
   - `faiss-cpu`: Vector similarity search
   - `scikit-learn`: Vector normalization
   - `python-dotenv`: Environment variable management
   - `requests`: API calls
   - `torch`: PyTorch (dependency for sentence-transformers)

3. **Prepare data files**

   Ensure `data/holdings.csv` and `data/trades.csv` are present in the data directory. The CSV files should contain the following columns:

   **holdings.csv:**
   - PortfolioName, ShortName, SecurityId, SecName, Qty, Price, MV_Base, PL_YTD, PL_MTD, PL_QTD, etc.

   **trades.csv:**
   - PortfolioName, TradeTypeName, SecurityId, Name, Ticker, Quantity, Price, Principal, TotalCash, etc.

4. **Verify directory structure**

   The `data/cache/` directory will be created automatically when you run the notebook for the first time.

---

## Configuration

### API Key Setup

**CRITICAL:** For security reasons, this repository does not include API keys. You must configure your own OpenRouter API key.

1. **Sign up for OpenRouter**
   - Visit https://openrouter.ai/
   - Create a free account
   - Generate an API key

2. **Select a free model**
   - OpenRouter offers several free models
   - Recommended: `openai/gpt-4o-mini` (used in this project)
   - Other free options available at https://openrouter.ai/models

3. **Create environment file**

   Copy the example file:
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and paste your API key:
   ```
   OPENROUTER_API_KEY=your_actual_api_key_here
   ```

4. **Verify configuration**

   When you run the notebook, you should see:
   ```
   API key loaded successfully
   ```

   If you see a warning, check that:
   - The `.env` file exists in the root directory
   - The API key is correctly pasted without extra spaces
   - The file is named exactly `.env` (not `.env.txt`)

---

## Usage

### Running the System

1. **Open the Jupyter notebook**
   ```bash
   jupyter notebook main.ipynb
   ```

2. **Execute cells in order**
   - Start from the top and run each cell sequentially
   - The first run will generate cache files (this may take a few minutes)
   - Subsequent runs will load from cache and be much faster

3. **Test with example queries**

   The notebook includes a test suite with 30 questions. Run the "TEST RUN" section to see results.

4. **Ask custom questions**

   Use the "CUSTOM QUERY RUN" section at the end:
   ```python
   query = "Total PL YTD for Wellington Portfolio?"
   answer = answer_question(query, debug=False)
   print(answer)
   ```

### Query Examples

**Count Operations:**
- "How many total Buy trades were executed across all portfolios?"
- "Total holdings for Garfield"

**Aggregations:**
- "Total PL YTD for Wellington Portfolio"
- "Total quantity of AA stock traded?"
- "Total MV Base for Redwood Income Fund"

**Rankings:**
- "Which fund has best quarterly profit?"
- "Show top 3 portfolios by market value"
- "Which fund has worst yearly PL?"

**Filtering:**
- "Show holdings with negative PL_YTD"
- "Show all trades for SecurityId 290081"

---

## Test Results

The system was tested with 30 diverse queries covering various operation types. Here are representative results:

### Successful Queries

**Q1:** "Which fund has best quarterly profit?"  
**A1:** The ranking by PL QTD shows that Ytum has the highest profit at $1,493,081.48, followed by Platpot at $1,489,647.46, Opium Holdings Partners at $513,265.37, NPSMF1 at $46,415.46, and NPSMF2 at $45,402.83.

**Q8:** "Total PL YTD for Garfield"  
**A8:** The total PL YTD for Garfield is $-168,551,028.29.

**Q10:** "Total quantity of AA stock traded?"  
**A10:** The total quantity of AA stock traded is 15,100.00.

**Q13:** "Show holdings with negative PL_YTD"  
**A13:** There are 303 holdings with negative PL YTD.

**Q22:** "What is the total MV Base for Heather?"  
**A22:** The total MV Base for Heather is $131,835,765.16.

### Queries with No Data Found

**Q3:** "Total trades for GreenRock Capital?"  
**A3:** Sorry can not find the answer.

**Q17:** "Total quantity of TSLA stock traded?"  
**A17:** Sorry can not find the answer.

These results indicate that the portfolio/ticker does not exist in the dataset.

---

## Technical Implementation

### Key Components

1. **Data Normalization**
   - Text normalization removes special characters and standardizes case
   - Numeric conversion handles various formats (commas, parentheses for negatives)
   - Missing values handled gracefully

2. **Entity Extraction**
   - Pattern-based matching for structured entities (tickers, metrics)
   - Fuzzy string matching (difflib) for portfolio names with 72% similarity threshold
   - Context-aware extraction (e.g., "buy" implies trade type)

3. **Semantic Search**
   - Model: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional embeddings)
   - Index: FAISS with Inner Product (cosine similarity via normalization)
   - Top-K retrieval with configurable minimum score threshold (0.35)

4. **Query Classification**
   - Dataset detection based on keywords (trades vs. holdings vs. both)
   - Operation type inference (count/aggregate/rank/show/unknown)
   - Handles ambiguous queries by trying multiple strategies

5. **Response Generation**
   - Strict LLM prompting to prevent hallucination
   - Evidence-based formatting (only uses computed summaries)
   - Fallback to simple formatting when API unavailable

### Configuration Parameters

```python
CHUNK_TOP_K = 5              # Number of chunks to retrieve
CHUNK_MIN_SCORE = 0.35       # Minimum similarity score
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # Sentence transformer model
```

These can be adjusted in the notebook for different performance/accuracy tradeoffs.

---

## Troubleshooting

**API Key Issues:**
- Verify `.env` file exists and contains correct key
- Check for extra spaces or newlines
- Ensure key is valid by testing at https://openrouter.ai/

**Performance Issues:**
- First run will be slow due to embedding generation
- Cache files should be preserved between runs
- If cache corruption occurs, delete `data/cache/` and rerun

**Query Not Working:**
- Enable debug mode: `answer_question(query, debug=True)`
- Check entity extraction output
- Verify portfolio/ticker names exist in data

**Missing Dependencies:**
- Reinstall requirements: `pip install -r requirements.txt --force-reinstall`
- Check Python version compatibility

---