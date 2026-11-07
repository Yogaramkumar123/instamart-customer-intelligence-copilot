# Instamart Customer Intelligence & Support Copilot

An enterprise-grade, reproducible data engineering and intelligence pipeline designed for quick-commerce operations across six metro cities in India. This repository delivers **Part A: Data Foundation — Clean, Store, Query** of the capstone project, turning raw, noisy transaction exports into a trusted, relational analytical database and answering core business intelligence questions through SQL and pandas.

---

## 1. Overview

### 1.1 Project Architecture
Quick-commerce environments like Swiggy Instamart process tens of thousands of orders across distributed dark stores within minutes. Operational data exports frequently suffer from data quality issues: duplicate records, partial duplicate pending states, negative delivery durations, corrupt basket amounts, non-standard categorical spellings, and anomalous customer demographics.

The **Instamart Customer Intelligence & Support Copilot** provides an end-to-end analytical foundation structured into four progressive stages:
- **Part A: Data Foundation (Clean, Store, Query)** — Ingestion, multi-table auditing, deterministic sequential data cleaning, relational SQLite data warehouse modeling, and analytical querying via SQL and pandas.
- **Part B: Customer Segmentation & Lapse Prediction** — RFM feature extraction, K-Means clustering, PCA dimensionality reduction, and leakage-free customer lapse prediction.
- **Part C: Support Ticket Triage with Guardrails** — PII masking, token budgeting, transient error recovery, and LLM-powered ticket classification.
- **Part D: Policy RAG Assistant & Refund Agent** — Deterministic chunking, TF-IDF vector retrieval over operational policy documents, grounded policy question answering, and an autonomous ReAct refund agent with persistent crash recovery.

```
instamart-copilot/
│
├── generate_data.py            # Verbatim synthetic dataset generator
├── mock_llm.py                 # Verbatim offline LLM test harness
├── requirements.txt            # Locked project dependencies
├── .gitignore                  # Git exclusion rules
├── README.md                   # System documentation and reports
│
├── src/
│   ├── __init__.py
│   ├── part_a/
│   │   ├── __init__.py
│   │   ├── audit.py            # Dataset structure and anomaly profiling
│   │   ├── clean_customers.py  # Customer demographic cleansing & imputation
│   │   ├── clean_products.py   # Product catalog validation
│   │   ├── clean_orders.py     # Sequential order cleaning & deduplication
│   │   ├── database.py         # Relational SQLite database pipeline
│   │   ├── sql_analysis.py     # Seven analytical business queries runner
│   │   ├── pandas_analysis.py  # Pivot tables, descriptive stats & correlations
│   │   └── validate.py         # Automated checkpoint validation suite
│   │
│   ├── part_b/
│   │   ├── __init__.py
│   │   ├── rfm_analysis.py     # Customer RFM feature calculation
│   │   ├── clustering.py       # K-Means clustering (K=2..8) and selection
│   │   ├── segment_profiling.py# Cluster profiling, naming & export
│   │   ├── pca_analysis.py     # 2D PCA projection & variance analysis
│   │   ├── lapse_features.py   # Leakage-free cohort & feature matrix
│   │   ├── preprocessing.py    # Stratified split & ColumnTransformer
│   │   ├── model_training.py   # Baseline, Logistic Regression & Random Forest tuning
│   │   ├── class_balance.py    # Training-only random oversampling experiment
│   │   ├── model_interpretation.py # Feature importance & coefficient readings
│   │   ├── retention_list.py   # Illustrative Premium retention ranking
│   │   └── validate.py         # Comprehensive Part B validation suite
│   │
│   ├── part_c/
│   │   ├── __init__.py
│   │   ├── guardrails.py       # Input token and empty ticket guardrails
│   │   ├── pii_redaction.py    # Sequential regex PII masking and verification
│   │   ├── prompts.py          # Zero-shot and few-shot system prompts
│   │   ├── retry_handler.py    # Exponential backoff with jitter on TransientLLMError
│   │   ├── classifier.py       # Ticket classifier executing prompt & validation
│   │   ├── output_validation.py# Strict JSON parsing & lowercase keyword extraction
│   │   ├── token_budget.py     # Cumulative token budget enforcement
│   │   ├── routing.py          # Segment-driven 4-tier ticket routing engine
│   │   ├── pipeline.py         # End-to-end triage pipeline & evaluation metrics
│   │   └── validate.py         # Comprehensive Part C validation suite
│   │
│   └── part_d/
│       ├── __init__.py
│       ├── document_loader.py  # Ingestion of the 5 policy documents
│       ├── chunking.py         # Deterministic 40-word sliding window chunker
│       ├── vector_store.py     # TF-IDF matrix and cosine similarity search
│       ├── retrieval.py        # Top-k chunk retrieval interface
│       ├── policy_assistant.py # Grounded policy question answerer & evaluation
│       ├── tools.py            # Five autonomous agent tool functions
│       ├── tool_registry.py    # Tool specifications schema and dispatch map
│       ├── refund_calculator.py# Deterministic policy-governed refund calculation
│       ├── agent.py            # ReAct agent loop with timeout and token ceilings
│       ├── checkpoint.py       # State persistence and snapshot manager
│       ├── resume.py           # Resumable execution loop from saved checkpoints
│       └── validate.py         # Comprehensive Part D validation suite
│
├── scripts/
│   ├── run_part_a.py           # End-to-end master orchestration pipeline (Part A)
│   ├── validate_part_a.py      # Standalone verification runner (Part A)
│   ├── run_part_b.py           # End-to-end master orchestration pipeline (Part B)
│   ├── validate_part_b.py      # Standalone verification runner (Part B)
│   ├── run_part_c.py           # End-to-end master orchestration pipeline (Part C)
│   ├── validate_part_c.py      # Standalone verification runner (Part C)
│   ├── run_part_d.py           # End-to-end master orchestration pipeline (Part D)
│   └── validate_part_d.py      # Standalone verification runner (Part D)
│
├── data/
│   ├── raw/                    # Raw generated CSV files and policies
│   ├── clean/                  # Cleaned CSV files (customers, products, orders)
│   ├── policies/               # Operational policy documents
│   └── instamart.db            # SQLite relational database
│
├── sql/
│   └── queries.sql             # Production SQL scripts (Q1 to Q7)
│
├── outputs/
│   ├── customer_segments.csv   # Customer segment profiles (Part B)
│   ├── lapse_model.joblib      # Tuned serialized lapse prediction pipeline (Part B)
│   ├── retention_list.csv      # Prioritized test-set Premium retention list (Part B)
│   ├── triage_results.csv      # Final 164-ticket production routing table (Part C)
│   ├── triage_budget_demo.csv  # Token budget cut-off demonstration (Part C)
│   ├── agent_runs.jsonl        # Seven ReAct scenario audit logs (Part D)
│   ├── refund_ledger.jsonl     # Transactional ledger of issued and escalated refunds (Part D)
│   ├── part_a/                 # Query outputs, pivot tables, audit logs
│   └── part_b/                 # K-Means comparisons and PCA summaries
│
└── state/                      # Persistent execution checkpoint snapshots
```

### 1.2 Technology Stack & Dependencies
- **Language**: Python 3.11+ (tested on Python 3.12.10)
- **Pinned Dependencies (`requirements.txt`)**:
  - `pandas==2.2.3`
  - `numpy==2.2.5`
  - `scikit-learn==1.6.1`
  - `joblib==1.4.2`
- **Database**: SQLite 3.25+ (bundled with standard Python, supporting full window functions and CTEs)
- **All Monetary Figures**: Expressed strictly in Indian Rupees (INR).

---

## 2. Reproduce

To reproduce the entire data foundation, relational database, analytical queries, and checkpoint verification from a clean repository clone, execute the following numbered commands in order:

1. **Create and activate an isolated virtual environment**:
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. **Install locked dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate synthetic data**:
   ```bash
   python generate_data.py
   ```
   *Expected console output:*
   `generated: customers=1200 products=40 orders_raw=13084 tickets=164`

4. **Run the complete Part A pipeline**:
   ```bash
   python scripts/run_part_a.py
   ```
   *Executes the audit, customer cleaning, product validation, sequential order cleaning, SQLite database creation, seven SQL queries, and pandas exploratory analysis.*

5. **Execute Part A standalone validation**:
   ```bash
   python scripts/validate_part_a.py
   ```
   *Output ends with the exact string:* `ALL CHECKPOINTS PASSED`

6. **Run the complete Part B pipeline**:
   ```bash
   python scripts/run_part_b.py
   ```
   *Executes RFM table construction, K-Means clustering, PCA projection, leakage-free feature engineering, model training and grid searches, class balancing, model persistence, and retention list generation.*

7. **Execute Part B standalone validation**:
   ```bash
   python scripts/validate_part_b.py
   ```
   *Output ends with the exact string:* `ALL PART B CHECKPOINTS PASSED`

8. **Run the complete Part C pipeline**:
   ```bash
   python scripts/run_part_c.py
   ```
   *Executes input guardrails, PII redaction audit, zero-shot and few-shot triage classifications with exponential backoff retry recovery, output validation, token budget demo (`outputs/triage_budget_demo.csv`), customer segment-driven ticket routing (`outputs/triage_results.csv`), and automated validation.*

9. **Execute Part C standalone validation**:
   ```bash
   python scripts/validate_part_c.py
   ```
   *Output ends with the exact string:* `ALL PART C CHECKPOINTS PASSED`

10. **Run the complete Part D pipeline**:
    ```bash
    python scripts/run_part_d.py
    ```
    *Executes policy document chunking, TF-IDF vector retrieval, RAG grounding evaluation, zero-overlap comparison, tool definitions, ReAct refund agent across all seven scenarios (`outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`), safety limits, crash recovery demonstration, and automated validation.*

11. **Execute Part D standalone validation**:
    ```bash
    python scripts/validate_part_d.py
    ```
    *Output ends with the exact string:*
    ```text
    ==================================================
    ALL PART D CHECKPOINTS PASSED
    ==================================================
    ```

---

## 3. Part A: Data Foundation — Clean, Store, Query

### 3.1 Raw Data Audit
The raw data exports represent transactions, customer profiles, catalog data, and incoming support tickets. Ticket data was ingested with `keep_default_na=False` to preserve empty text fields as empty strings.

#### Table Shapes & Missingness Summary
| Table | Rows | Columns | Column Details (Data Type, Missing Count, Missing %) |
|---|---:|---:|---|
| `customers.csv` | 1,200 | 5 | `customer_id` (object, 0, 0.0%), `city` (object, 0, 0.0%), `signup_date` (object, 0, 0.0%), `age` (float64, 61, 5.08%), `gender` (object, 30, 2.50%) |
| `products.csv` | 40 | 4 | `product_id` (object, 0, 0.0%), `product_name` (object, 0, 0.0%), `category` (object, 0, 0.0%), `price_inr` (int64, 0, 0.0%) |
| `orders.csv` | 13,084 | 9 | `order_id` (object, 0, 0.0%), `customer_id` (object, 0, 0.0%), `product_id` (object, 0, 0.0%), `quantity` (int64, 0, 0.0%), `order_ts` (object, 0, 0.0%), `status` (object, 0, 0.0%), `discount_pct` (int64, 0, 0.0%), `order_amount` (object, 53, 0.41%), `delivery_minutes` (float64, 2,541, 19.42%) |
| `tickets.csv` | 164 | 5 | `ticket_id` (object, 0, 0.0%), `customer_id` (object, 0, 0.0%), `order_id` (object, 0, 0.0%), `text` (object, 0, 0.0%), `true_category` (object, 0, 0.0%) |

#### Specific Audit Findings
| Metric | Expected Reference | Actual Measured | Status |
|---|---:|---:|:---:|
| Raw order rows | 13,084 | 13,084 | MATCH |
| Exact duplicate order rows | 119 | 119 | MATCH |
| Pending order rows | 656 | 656 | MATCH |
| Distinct order IDs | 12,309 | 12,309 | MATCH |
| Non-numeric order amounts | 120 | 120 | MATCH |
| Negative delivery times | 106 | 106 | MATCH |
| Blank delivery times | 2,541 | 2,541 | MATCH |
| Blank ages (`customers.csv`) | 61 | 61 | MATCH |
| Ages outside 10–100 (`customers.csv`) | 6 | 6 | MATCH |
| Distinct raw city spellings (`customers.csv`) | 24 | 24 | MATCH |
| Distinct non-blank raw gender spellings | 10 | 10 | MATCH |
| Blank genders (`customers.csv`) | 30 | 30 | MATCH |

---

### 3.2 Customer Cleaning (`src/part_a/clean_customers.py`)

#### Implemented Cleaning Rules
1. **City Standardization**: Stripped leading/trailing whitespace and applied title-casing (`.str.strip().str.title()`). Changed **476** non-standard text values, collapsing 24 raw variations into exactly six canonical cities.
2. **Gender Normalization**: Mapped all 10 spelling variations to canonical categories (`male`, `female`, `unknown`). Converted **30** blank/null values to `unknown`. Total gender values modified: **950**.
3. **Age Validation**: Identified **6** impossible age values outside [10, 100] (values: 280, -4, 0, 150, 999, 200) and treated them as missing (`np.nan`), bringing total missing ages to **67** (61 initially blank + 6 out-of-bounds).
4. **City-Level Median Imputation**: Imputed all **67** missing ages using the median age of the customer's corresponding city via `groupby('city')['age'].transform('median')`.
5. **Signup Date Formatting**: Converted `signup_date` to standard datetime and serialized as `YYYY-MM-DD`.

#### Cleaned Customer Distributions
- **Cleaned City Counts**:
  - Pune: **227**
  - Delhi: **210**
  - Bengaluru: **197**
  - Chennai: **191**
  - Hyderabad: **191**
  - Mumbai: **184**
  - *Total*: **1,200** (Exactly 6 cities)
- **Cleaned Gender Counts**:
  - female: **610**
  - male: **560**
  - unknown: **30**
  - *Total*: **1,200**
- **Cleaned Age Summary**:
  - Missing ages: **0**
  - Minimum age: **18**
  - Maximum age: **60**
  - *Result exported to*: `data/clean/customers.csv`

---

### 3.3 Product Cleaning (`src/part_a/clean_products.py`)
- Verified catalog structure: preserved `product_id`, `product_name`, `category`, `price_inr`.
- Validated **40** unique product IDs (`P001` to `P040`).
- Validated categories against authorized set: `Grocery`, `Dairy`, `Snacks`, `Personal Care`, `Electronics`.
- Validated that all prices are strictly positive (Minimum: INR 20, Maximum: INR 1,999).
- *Result exported to*: `data/clean/products.csv`.

---

### 3.4 Order Cleaning (`src/part_a/clean_orders.py`)

Cleaning was executed in the strictly prescribed sequence:

1. **Timestamp Conversion**: Converted `order_ts` to datetime format.
2. **Exact Deduplication**: Removed **119** exact duplicate rows, reducing orders from 13,084 to 12,965 rows.
3. **Latest Record Retention by Order ID (Deterministic Whole-Row Selection)**: Sorted orders chronologically by `order_ts` and applied `.drop_duplicates(subset="order_id", keep="last")` to preserve the complete latest record for each order. This removed **656** earlier pending records, eliminating all pending rows and retaining the final status. Row count reduced to exactly **12,309**.
4. **Order Amount Recovery**: Converted `order_amount` to numeric with error coercion. Recovered **112** missing/unparseable values in the deduplicated table using catalog pricing:
   $$\text{order\_amount} = \text{round}\left(\text{price\_inr} \times \text{quantity} \times \left(1 - \frac{\text{discount\_pct}}{100}\right), 2\right)$$
   Missing amounts remaining: **0**.
5. **Negative Delivery Times Coercion**: Converted **105** negative delivery durations in the deduplicated table to `np.nan`.
6. **Completed Orders Delivery Imputation**: Imputed missing delivery times for completed orders using the category median delivery time among completed orders via `groupby('category')['delivery_minutes'].transform('median')`. Imputed **774** records; missing delivery times in completed orders remaining: **0**.
7. **Cancelled Orders Retention**: Maintained all **1,199** cancelled orders with missing (`NaN`) delivery minutes. No delivery time was invented for orders that were never delivered.
8. **Export**: Stored `order_ts` as `YYYY-MM-DD HH:MM:SS` and exported clean orders to `data/clean/orders.csv`.

#### Final Order Cleaning Checkpoints
| Metric | Expected Reference | Actual Cleaned | Status |
|---|---:|---:|:---:|
| Total orders | 12,309 | 12,309 | MATCH |
| Completed orders | 11,110 | 11,110 | MATCH |
| Cancelled orders | 1,199 | 1,199 | MATCH |
| Pending orders | 0 | 0 | MATCH |
| Missing order amounts | 0 | 0 | MATCH |
| Missing delivery times (completed orders) | 0 | 0 | MATCH |
| Missing delivery times (cancelled orders) | 1,199 | 1,199 | MATCH |
| Completed order revenue | INR 2,998,668.80 | INR 2,998,668.80 | MATCH |

---

### 3.5 SQLite Database (`src/part_a/database.py`)
Created SQLite relational database at `data/instamart.db` with enforced relational schema, foreign key constraints, and performance indexes:

```sql
CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY,
    city TEXT NOT NULL,
    signup_date TEXT NOT NULL,
    age INTEGER NOT NULL,
    gender TEXT NOT NULL
);

CREATE TABLE products (
    product_id TEXT PRIMARY KEY,
    product_name TEXT NOT NULL,
    category TEXT NOT NULL,
    price_inr REAL NOT NULL
);

CREATE TABLE orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    order_ts TEXT NOT NULL,
    status TEXT NOT NULL,
    discount_pct REAL NOT NULL,
    order_amount REAL NOT NULL,
    delivery_minutes REAL,
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id),
    FOREIGN KEY (product_id) REFERENCES products (product_id)
);
```

#### Row Count Validation
- `customers`: **1,200** rows inserted.
- `products`: **40** rows inserted.
- `orders`: **12,309** rows inserted.

---

### 3.6 Seven SQL Analytical Queries (`sql/queries.sql`)
Executed using `pd.read_sql_query()` against `data/instamart.db`. Output CSVs saved in `outputs/part_a/`.

#### Summary of Query Results
| Query ID | Description | Key Reference Target | Actual Measured Result | Status |
|---|---|---|---|:---:|
| **Q1** | Completed orders > INR 500, customer total > INR 8,000 (WHERE & HAVING) | 30 customers | **30** customers | MATCH |
| **Q2** | 3-table JOIN: Revenue by city and product category | 30 rows; Grand Total INR 2,998,668.80; Bengaluru/Electronics INR 233,909.45 | **30** rows; Total **INR 2,998,668.80**; Bengaluru/Electronics = **INR 233,909.45** | MATCH |
| **Q3** | Scalar subquery: Products priced above catalogue average | 11 products | **11** products | MATCH |
| **Q4** | LEFT JOIN: Customers without a completed order | 25 customers | **25** customers | MATCH |
| **Q5** | CTE: Customers with completed spend above average customer spend | 550 customers; Average spend INR 2,552.06 | **550** customers; Average spend **INR 2,552.06** | MATCH |
| **Q6** | Window functions: RANK(), DENSE_RANK(), ROW_NUMBER() per city | 6 top customers; exactly 8 rows where RANK() != ROW_NUMBER() | Top per city matched; exactly **8** rows differ | MATCH |
| **Q7** | Running total: Daily and cumulative completed revenue | 183 daily rows; Final cumulative INR 2,998,668.80 | **183** rows; Final cumulative **INR 2,998,668.80** | MATCH |

#### Detailed SQL Analysis Findings

##### Q6: Top Customer per City & Ranking Discrepancy
- **Bengaluru**: Customer `C0082` — Spend: **INR 12,856.20**
- **Chennai**: Customer `C0930` — Spend: **INR 10,151.40**
- **Delhi**: Customer `C0291` — Spend: **INR 13,742.60**
- **Hyderabad**: Customer `C0743` — Spend: **INR 12,730.05**
- **Mumbai**: Customer `C0693` — Spend: **INR 14,766.55**
- **Pune**: Customer `C0426` — Spend: **INR 10,377.10**

> [!NOTE]
> **Why ties cause RANK() to differ from ROW_NUMBER():**
> When multiple customers within a city have identical total completed spend, `RANK()` assigns them the identical rank number (skipping subsequent rank positions by the tie count), whereas `ROW_NUMBER()` assigns strictly sequential, distinct integers to every row regardless of ties.

---

### 3.7 Pandas Cross-Check & Exploratory Analysis (`src/part_a/pandas_analysis.py`)

#### Pivot Table Cross-Check (Rebuilding Q2)
Rebuilt using `pd.pivot_table(df_merged, values='order_amount', index='city', columns='category', aggfunc='sum', margins=True)`.
- **Pivot Table Grand Total**: **INR 2,998,668.80** (Exact match with SQL Q2 total).

#### Per-City Completed Order Aggregations
| City | Completed Spend (INR) | Mean Order Value (INR) | Completed Order Count |
|---|---:|---:|---:|
| **Bengaluru** | 523,643.40 | 276.47 | 1,894 |
| **Chennai** | 463,629.15 | 272.72 | 1,700 |
| **Delhi** | 502,829.80 | 243.62 | 2,064 |
| **Hyderabad** | 528,589.90 | 284.95 | 1,855 |
| **Mumbai** | 479,058.50 | 297.00 | 1,613 |
| **Pune** | 500,918.05 | 252.48 | 1,984 |
| **Total** | **2,998,668.80** | **269.91** | **11,110** |

#### Completed Order Amount Descriptive Statistics & Outliers
| Metric | Reference Value | Actual Measured |
|---|---:|---:|
| **Mean** | INR 269.91 | INR 269.91 |
| **Median** | INR 135.00 | INR 135.00 |
| **Skewness** | ≈ 4.61 | 4.6088 |
| **Q1 (25th Percentile)** | INR 64.80 | INR 64.80 |
| **Q3 (75th Percentile)** | INR 297.00 | INR 297.00 |
| **Interquartile Range (IQR)** | INR 232.20 | INR 232.20 |
| **Lower Fence (Q1 - 1.5×IQR)** | INR -283.50 | INR -283.50 |
| **Upper Fence (Q3 + 1.5×IQR)** | INR 645.30 | INR 645.30 |
| **Lower Outliers (< Lower Fence)** | 0 | **0** |
| **Upper Outliers (> Upper Fence)** | 897 | **897** |

#### Correlation Matrix
| Feature | `order_amount` | `discount_pct` | `quantity` | `delivery_minutes` |
|---|---:|---:|---:|---:|
| `order_amount` | 1.0000 | -0.0951 | 0.0873 | -0.0031 |
| `discount_pct` | -0.0951 | 1.0000 | -0.0004 | -0.0098 |
| `quantity` | 0.0873 | -0.0004 | 1.0000 | -0.0126 |
| `delivery_minutes` | -0.0031 | -0.0098 | -0.0126 | 1.0000 |

- **Largest Absolute Correlation**: `order_amount` and `discount_pct` at **-0.0951** (followed closely by `order_amount` and `quantity` at **+0.0873**).

#### Synthesis & Interpretation (Word Count: 147 words)
The dataset captures 11,110 completed grocery and retail quick-commerce deliveries across six Indian metros between April and September 2025. Order amounts average INR 269.91 with a median of INR 135.00, demonstrating substantial right-skewness (skewness ≈ 4.61) driven by high-ticket electronics and bundled personal-care baskets. Exactly 897 upper outliers exceed the INR 645.30 threshold (Q3 + 1.5×IQR), while zero lower outliers exist. Pairwise correlation analysis reveals the strongest association between order_amount and discount_pct (r ≈ -0.095), followed by order_amount and quantity (r ≈ +0.087). These negligible correlations indicate that basket value varies primarily by catalog item price rather than coupon depth or unit count. Importantly, correlation of this small magnitude does not establish causation, as promotions and order basket sizes reflect multi-factor consumer purchasing decisions.

---

### 3.8 Automated Validation Status
Executing `python scripts/validate_part_a.py` executes **59 runtime checks** across file existence, schema validity, demographic distributions, order integrity, revenue consistency, SQL query results, and statistical distributions.

```
==================================================
STARTING PART A COMPREHENSIVE VALIDATION
==================================================
  [PASS] Clean Customers CSV Exists: True
  [PASS] Clean Products CSV Exists: True
  [PASS] Clean Orders CSV Exists: True
  [PASS] SQLite Database Exists: True
  [PASS] SQL Queries File Exists: True
  ...
  [PASS] Q7 Daily Rows Count: 183
  [PASS] Q7 Final Cumulative Revenue (INR): 2998668.8
  [PASS] Completed Amount Upper Fence (INR): 645.3
  [PASS] Upper Outliers Count: 897
  [PASS] Top Correlation Value: -0.0951
==================================================
ALL CHECKPOINTS PASSED
==================================================
```

---

## 4. Part B: Customer Segments and Lapse Prediction

### 4.1 RFM Table Formulation (`src/part_b/rfm_analysis.py`)
Customer segmentation utilizes whole-window completed transaction history from `data/clean/orders.csv`. As of the fixed snapshot moment **2025-10-01 00:00:00**, three behavioral metrics were calculated for all 1,175 customers with at least one completed order:
- **Recency**: Whole days elapsed between the snapshot moment and the customer's most recent completed order (`(snapshot - max(order_ts)).dt.days`).
- **Frequency**: Total number of distinct completed orders.
- **Monetary**: Total lifetime sum of completed order amounts in Indian Rupees (INR).

#### RFM Verification Checkpoints
| Metric | Reference Target | Actual Measured | Status |
|---|---:|---:|:---:|
| Qualifying Active Customers | 1,175 | **1,175** | MATCH |
| Recency Median | 19 days | **19.0 days** | MATCH |
| Recency Minimum | 0 days | **0 days** | MATCH |
| Recency Maximum | 180 days | **180 days** | MATCH |
| Frequency Mean | 9.46 | **9.46** | MATCH |
| Frequency Maximum | 33 | **33** | MATCH |
| Monetary Mean | INR 2,552.06 | **INR 2,552.06** | MATCH |
| Monetary Maximum | INR 14,766.55 | **INR 14,766.55** | MATCH |

---

### 4.2 K-Means Clustering & K Selection (`src/part_b/clustering.py`)
RFM features were standardized using `StandardScaler` to ensure scale invariance across dimensions with disparate physical units (days, count, INR). `KMeans(n_clusters=K, n_init=10, random_state=42)` was fitted for $K \in [2, 8]$.

#### Clustering Evaluation Metrics
| K | Inertia (Target ±5%) | Actual Inertia | Silhouette Score (Target ±0.05) | Actual Silhouette | Status |
|---|---:|---:|---:|---:|:---:|
| 2 | 1,820.8 | **1,820.8** | 0.465 | **0.465** | MATCH |
| 3 | 1,268.2 | **1,268.2** | 0.445 | **0.445** | MATCH |
| **4** | **743.1** | **743.1** | **0.526** | **0.526** | **OPTIMAL** |
| 5 | 617.4 | **617.4** | 0.481 | **0.480** | MATCH |
| 6 | 503.2 | **503.2** | 0.482 | **0.482** | MATCH |
| 7 | 408.5 | **408.5** | 0.416 | **0.416** | MATCH |
| 8 | 366.5 | **366.5** | 0.412 | **0.412** | MATCH |

#### Justification for Selecting $K = 4$
1. **Silhouette Score Peak**: $K=4$ achieves the global maximum silhouette score of **0.526** across all tested cluster counts, demonstrating optimal cluster cohesion and inter-cluster separation.
2. **Distinct Elbow Drop**: The reduction in inertia from $K=3$ (1,268.2) to $K=4$ (743.1) represents a sharp marginal decrease of **41.4%**, after which gains diminish significantly ($K=5$ drops only 16.9% to 617.4).
3. **Business Actionability**: Four clusters produce intuitively distinct customer behavioral archetypes that the retention marketing team can target with tailored retention playbooks (Champions, High-Value Spenders, Regulars, and At-Risk shoppers).

> [!NOTE]
> **Why K=8 has lower inertia than K=2 yet cannot determine K:**
> Inertia measures the within-cluster sum of squared errors (SSE). As the number of centroids $K$ increases, data points are geometrically closer to their nearest centroid, causing inertia to decrease monotonically until $K=N$ (where inertia equals zero). Minimizing inertia alone leads to trivial overfitting; optimal cluster selection requires balancing compression error against cluster separation (silhouette coefficient) and business interpretability.

---

### 4.3 Customer Segment Profiling (`src/part_b/segment_profiling.py`)
Cluster IDs are arbitrary categorical labels without intrinsic ordinal ranking. The four identified segments were profiled and assigned operational names and retention actions based on their centroids:

#### Segment Behavioral Profiles & Strategic Actions
| Segment ID | Segment Name | Size | Mean Recency | Mean Frequency | Mean Monetary (INR) | Premium Flag | Recommended Marketing & Retention Action |
|---|---|---:|---:|---:|---:|:---:|---|
| **0** | **Occasional Regulars** | 436 | 28.9 days | 3.8 | INR 1,057.62 | `False` | Mid-week replenishment vouchers, category discovery cross-sells, and basket-building coupons. |
| **1** | **Premium High-Spenders** | 138 | 24.1 days | 6.8 | **INR 7,129.70** | `True` | VIP support desk routing, early access to new launches, and concierge delivery guarantees. |
| **2** | **Frequent Champions** | 419 | 8.8 days | 19.4 | INR 3,410.02 | `False` | Loyalty milestone multipliers, automated subscription replenishment, and referral bonuses. |
| **3** | **At-Risk / Hibernating** | 182 | 109.0 days | 2.2 | INR 685.99 | `False` | Win-back discount blitzes, satisfaction diagnostic surveys, and multi-channel push re-engagement. |

- **Premium Assignment**: Exactly one cluster (Segment 1, 138 customers) exhibits the highest mean monetary value (INR 7,129.70) and is designated `is_premium = True`.
- **Export**: Saved to `outputs/customer_segments.csv` with columns: `customer_id`, `recency`, `frequency`, `monetary`, `segment_id`, `segment_name`, `is_premium`.

---

### 4.4 PCA Dimensionality Reduction (`src/part_b/pca_analysis.py`)
Applied `PCA(n_components=2)` to the standardized RFM matrix to evaluate linear variance retention in 2D space:
- **PC1 Explained Variance Ratio**: **0.6343** (Reference: ~0.634, within $\pm 0.02$)
- **PC2 Explained Variance Ratio**: **0.2180** (Reference: ~0.218, within $\pm 0.02$)
- **Total Explained Variance**: **0.8523** (Reference: ~0.852, within $\pm 0.02$)
- *Summary saved to*: `outputs/part_b/pca_summary.json`.

---

### 4.5 Lapse Prediction Problem Formulation (`src/part_b/lapse_features.py`)
- **Temporal Cutoff**: **2025-09-01 00:00:00**.
- **Observation Period (Features)**: 2025-04-01 through 2025-08-31.
- **Outcome Period (Labels)**: 2025-09-01 through 2025-09-30.
- **Eligible Cohort**: Every customer with at least one completed order strictly before 2025-09-01 (**1,156** customers).
- **Target Definition**: `lapsed = 1` if the customer placed zero completed orders in September 2025; `lapsed = 0` if they placed $\ge 1$ completed order.
  - Cohort Size: **1,156** customers
  - Lapsed Customers: **443** (**38.32%** lapse rate)

#### Strict Leakage Prevention Architecture
1. **Zero Future Information**: No transaction or event dated on or after 2025-09-01 was permitted to enter feature calculations.
2. **Exclusion of Cluster Labels**: The RFM segment IDs and `is_premium` flags from Task 3 were strictly excluded from model feature matrices, as they incorporate whole-window September transaction data.
3. **Partition-Only Preprocessing**: All scalers and encoders were fitted strictly on training data (`X_train`) and merely applied to test rows.

#### Feature Definitions (Computed Strictly Pre-Cutoff)
- **Numeric**:
  - `recency`: Days between 2025-09-01 and customer's latest pre-cutoff completed order.
  - `frequency`: Total count of distinct completed orders before 2025-09-01.
  - `monetary`: Total sum of completed order amounts before 2025-09-01.
  - `average_discount_pct`: Mean discount percentage across pre-cutoff completed orders.
  - `cancellation_rate`: Cancelled orders divided by all finalized orders before 2025-09-01.
  - `average_delivery_minutes`: Mean delivery time among pre-cutoff completed orders.
  - `age`: Customer age from demographic database.
  - `tenure_days`: Whole days between customer `signup_date` and 2025-09-01.
- **Categorical**:
  - `favourite_category`: Most frequent product category among pre-cutoff completed orders.
  - `city`: Standardized metro city from demographic database.
  - `gender`: Normalized customer gender (`female`, `male`, `unknown`).

---

### 4.6 Preprocessing & Partitioning (`src/part_b/preprocessing.py`)
Partitioned using `train_test_split(test_size=0.2, stratify=y, random_state=42)`:
- **Training Set**: **924** customers (354 lapsed, 570 not lapsed)
- **Test Set**: **232** customers (89 lapsed, 143 not lapsed)

#### Demonstrating Preprocessor Training-Only Fit
- **Full Cohort Frequency Mean**: **8.0285**
- **Training-Fitted Scaler Frequency Mean**: **8.1147**
- *Explanation*: The training scaler mean reflects only the 924 training observations. Because the 80% stratified sample has a slightly different empirical mean than the 1,156 full cohort, the values diverge. This programmatic check verifies that test rows never leaked into the scaler.

---

### 4.7 Model Tuning & Evaluation (`src/part_b/model_training.py`)

#### Baseline Model
A naive baseline predicting "not lapsed" (`0`) for all customers yields:
- **Accuracy**: **61.64%** ($143 / 232$)
- **Precision**: **0.0000** (`zero_division=0`)
- **Recall**: **0.0000**
- **Lapsed Class F1**: **0.0000**

#### Hyperparameter Tuning (5-Fold Stratified Cross-Validation)
1. **Logistic Regression**: Evaluated over $C \in \{0.01, 0.1, 1, 10, 100\}$.
   - Best Parameter: `C = 100` (Best CV F1: **0.6506**)
2. **Random Forest**: Evaluated over `max_depth` $\in \{3, 5, 8\}$ and `min_samples_leaf` $\in \{5, 10, 20\}$ (45 fits total).
   - Best Parameters: `max_depth = 8`, `min_samples_leaf = 10` (Best CV F1: **0.6856**)

#### Performance on Untouched Test Set (Positive Lapsed Class)
| Model | Optimal Parameters | CV F1 | Test Accuracy | Test Precision | Test Recall | Test F1 |
|---|---|---:|---:|---:|---:|---:|
| **Baseline (Majority Class)** | N/A | N/A | 0.6164 | 0.0000 | 0.0000 | 0.0000 |
| **Logistic Regression** | `C=100` | 0.6506 | 0.7284 | 0.6444 | 0.6517 | **0.6480** |
| **Random Forest (Selected)** | `max_depth=8, min_samples_leaf=10` | **0.6856** | 0.7241 | 0.6404 | 0.6404 | **0.6404** |

#### Test Set Confusion Matrices
- **Logistic Regression**:
  $$\begin{bmatrix} 111 & 32 \\ 31 & 58 \end{bmatrix}$$
- **Random Forest**:
  $$\begin{bmatrix} 111 & 32 \\ 32 & 57 \end{bmatrix}$$

#### Model Selection Justification
**Random Forest** is selected as the primary production model:
1. **Higher Cross-Validation Generalization**: Random Forest achieved superior 5-fold CV performance (CV F1: **0.6856** vs **0.6506**), demonstrating better resilience to non-linear interactions across delivery latency, discount depth, and order recency.
2. **Cost-Asymmetric Business Trade-Off**: In quick-commerce retention, missing a lapsing customer (False Negative) is substantially more damaging than wrongly targeting an active customer (False Positive). Losing an active customer forfeits long-term Customer Lifetime Value (CLV) and requires high Customer Acquisition Cost (CAC) to replace. Conversely, an unnecessary win-back promotion sent to an active customer carries minimal friction. Both tuned models comfortably exceed the required benchmark of $F1 \ge 0.55$.

---

### 4.8 Class-Balance Experiment (`src/part_b/class_balance.py`)
The training split is imbalanced (354 lapsed vs 570 not lapsed). Random oversampling with replacement was applied exclusively to the training split until both classes reached **570** rows (1,140 total). The best Logistic Regression was refitted on the balanced data and evaluated against the unchanged test partition:

| Model Configuration | Training Split Class Counts | Test Precision | Test Recall | Test F1 |
|---|---|---:|---:|---:|
| **Original (Unbalanced) LR** | 354 lapsed / 570 not lapsed | **0.6444** | 0.6517 | 0.6480 |
| **Balanced (Oversampled) LR** | 570 lapsed / 570 not lapsed | 0.5891 | **0.8539** | **0.6972** |
| **Net Change** | $+216$ minority synthetic rows | $-0.0553$ | **$+0.2022$** | $+0.0492$ |

- **Key Takeaway**: **Recall shifted most**, surging by **+20.22 percentage points** (from 65.17% to 85.39%) while precision remained near 0.59. For Instamart's retention desk, this trade-off is highly advantageous: the balanced model intercepts **85% of lapsing customers** before they churn, at the cost of a modest increase in outreach volume.

---

### 4.9 Model Interpretation (`src/part_b/model_interpretation.py`)

#### Random Forest Top 5 Feature Importances
| Feature | Importance | Business Reading |
|---|---:|---|
| `monetary` | **0.2779** | Total pre-cutoff spend reflects historical customer value; drop-offs among higher spenders heavily shift tree boundaries. |
| `frequency` | **0.2769** | Order frequency captures habitual platform dependency; customers with entrenched weekly habits are far less likely to lapse. |
| `recency` | **0.1112** | Days since last order directly marks dormancy; extended inactivity signals impending churn. |
| `average_delivery_minutes` | **0.0617** | Delivery delays directly erode customer satisfaction; chronic delays trigger platform abandonment. |
| `average_discount_pct` | **0.0532** | High discount dependency flags deal-seekers vulnerable to competitor couponing. |

#### Logistic Regression Feature Coefficients
| Feature | Coefficient | Sign | Business Reading |
|---|---:|:---:|---|
| `frequency` | **-1.6982** | Negative | Strong negative driver: each unit increase in order frequency dramatically reduces the log-odds of lapsing. |
| `favourite_category_Electronics` | **-1.1367** | Negative | Electronics buyers demonstrate higher platform loyalty and substantially lower lapse probabilities. |
| `favourite_category_Grocery` | **-0.4489** | Negative | Regular grocery buyers exhibit sticky pantry-replenishment behavior. |
| `city_Delhi` | **-0.3744** | Negative | Customers in Delhi show higher retention rates relative to the baseline metro. |

---

### 4.10 Model Persistence (`outputs/lapse_model.joblib`)
The selected production pipeline (incorporating the full `ColumnTransformer` preprocessing and tuned `RandomForestClassifier`) was persisted to disk using `joblib.dump()` at `outputs/lapse_model.joblib`. Upon reloading with `joblib.load()`, out-of-sample predictions on `X_test` matched the original predictions with **100% exact parity**.

---

### 4.11 Illustrative Premium Retention List (`src/part_b/retention_list.py`)
Using the persisted model's predicted lapse probabilities, test-set customers identified as belonging to the Premium segment were ranked in descending order of churn risk. The top 10 candidates were exported to `outputs/retention_list.csv`:

#### Prioritized Test-Set Premium Retention Candidates
| customer_id | Predicted Lapse Probability | Actually Lapsed |
|---|---:|:---:|
| `C0862` | 0.5304 | 0 |
| `C1008` | 0.4775 | 0 |
| `C0060` | 0.4765 | 0 |
| `C0267` | 0.4753 | 0 |
| `C0133` | 0.4726 | 0 |
| `C0942` | 0.4706 | 0 |
| `C0553` | 0.4645 | 0 |
| `C1112` | 0.4620 | 0 |
| `C0565` | 0.4616 | **1** |
| `C1000` | 0.4602 | 0 |

> [!WARNING]
> **Post-Hoc Whole-Window Leakage Disclosure**:
> The `is_premium` flag was generated from whole-window RFM analysis (spanning April through September), meaning the cluster label saw transaction data from the same September window used to evaluate the lapse target. Consequently, this retention list serves as an **illustrative operational prioritization** rather than a leakage-free predictive evaluation. The model's own feature inputs strictly obeyed temporal boundaries and used only pre-cutoff orders.

---

### 4.12 Part B Automated Validation Status
Executing `python scripts/validate_part_b.py` executes 42 rigorous assertions across RFM checkpoints, K-Means clustering metrics, segment profiles, PCA ratios, feature leakage guards, model training, persistence, and retention list schema.

```
==================================================
STARTING PART B COMPREHENSIVE VALIDATION
==================================================
  [PASS] RFM Customer Count: 1175
  [PASS] Recency Median: 19.0
  [PASS] Recency Minimum: 0
  [PASS] Recency Maximum: 180
  [PASS] Frequency Mean: 9.46
  [PASS] Frequency Maximum: 33
  [PASS] Monetary Mean (INR): 2552.06
  [PASS] Monetary Maximum (INR): 14766.55
  [PASS] K=2 Inertia: 1820.8
  [PASS] K=3 Inertia: 1268.2
  [PASS] K=4 Inertia: 743.1
  [PASS] K=4 Silhouette: 0.526
  [PASS] K=5 Inertia: 617.4
  [PASS] Customer Segments Row Count: 1175
  [PASS] Unique Segment Count: 4
  [PASS] Exactly One Premium Cluster: 1
  [PASS] Premium Segment Size (~138): 138
  [PASS] Premium Segment Mean Monetary (~7130): 7129.7
  [PASS] PCA PC1 Variance Ratio: 0.6343
  [PASS] PCA PC2 Variance Ratio: 0.218
  [PASS] PCA Total Variance Ratio: 0.8523
  [PASS] Cohort Size: 1156
  [PASS] Lapsed Count: 443
  [PASS] Lapse Rate (%): 38.32
  [PASS] No Missing Values in Features: 0
  [PASS] Segment ID Not in Features: False
  [PASS] Train Rows Count: 924
  [PASS] Test Rows Count: 232
  [PASS] LR Test F1 >= 0.55: True
  [PASS] RF Test F1 >= 0.55: True
  [PASS] Balanced Training Class Counts: [570, 570]
  [PASS] Balanced Recall Jump (> 0.15 increase): True
  [PASS] Model File Exists: True
  [PASS] Reloaded Model Predictions Match Original: True
  [PASS] Retention List File Exists: True
  [PASS] Retention List Columns: ['customer_id', 'lapse_probability', 'actually_lapsed']
  [PASS] Retention List Probabilities Sorted Descending: True
==================================================
ALL PART B CHECKPOINTS PASSED
==================================================
```

---

## 5. Part C: Support Ticket Triage with Guardrails

### 5.1 Architecture & Pipeline Design
Customer support operations at Swiggy Instamart receive unstructured tickets ranging from urgent fraud reports and delivery inquiries to corrupted blanks and abusive/bloated inputs. Sending unvalidated, unredacted raw customer messages directly to an external LLM risks severe privacy breaches (PII leakage), runaway API costs, unhandled service outages, and hallucinated classifications.

Part C implements an enterprise-grade, offline support ticket triage pipeline wrapped with rigorous safety, resilience, and business intelligence guardrails:

```
Raw Ticket (data/raw/tickets.csv)
       │
       ▼
[Input Guardrails] ─── Empty / Whitespace? ──────► status="skipped_empty" (route: human_review, NORMAL)
       │           └── > 400 est_tokens?   ──────► status="flagged_long"  (route: human_review, NORMAL)
       ▼ (156 eligible tickets)
[PII Redaction] (Email -> [EMAIL], Phone -> [PHONE], Order ID -> [ORDER_ID])
       │
       ▼
[Token Budget Check] ── Total Tokens >= Budget? ──► status="budget_exhausted" (route: human_review, NORMAL)
       │
       ▼
[Resilient Mock LLM Call] (Exponential Backoff + Jitter on TransientLLMError, Max 3 Retries)
       │
       ▼
[Output Validation] ─── Valid JSON + Allowed Category? ──► status="classified_json"
       │            ─── Non-JSON + 1 Allowed Keyword?  ──► status="classified_keyword"
       │            ─── Invalid Category / Malformed?  ──► status="flagged_invalid_output" (human_review, NORMAL)
       ▼
[4-Tier Customer Segment Routing] (Reads outputs/customer_segments.csv from Part B)
       ├── Fraud Category                 ──► route: security_team, priority: HIGH
       ├── is_premium == True             ──► route: premium_desk,  priority: HIGH
       └── Standard Classified Ticket     ──► route: support_queue, priority: NORMAL
```

---

### 5.2 Input Guardrails (`src/part_c/guardrails.py`)
Incoming tickets are evaluated against two pre-model boundary conditions before any prompt assembly or network/model interaction:
1. **Empty / Whitespace Filter**: Tickets containing empty text or pure whitespace are immediately flagged as `skipped_empty` and never dispatched to the model.
2. **Token Ceiling Guard**: Ticket token length is estimated using the capstone standard formula:
   $$\text{est\_tokens}(text) = \lceil \text{word\_count} \times 1.3 \rceil$$
   Any ticket exceeding `MAX_INPUT_TOKENS = 400` is flagged as `flagged_long` and intercepted before model invocation.

#### Input Guardrail Audit Checkpoints
| Metric | Count | Checkpoint Reference | Audit Finding |
|---|---:|---:|---|
| Total Raw Tickets | 164 | 164 | Complete raw ticket corpus ingested with `keep_default_na=False`. |
| Empty / Whitespace Tickets (`skipped_empty`) | **4** | 4 | Tickets `T0157`, `T0158`, `T0159`, `T0160` intercepted without model call. |
| Over-length Tickets (`flagged_long`) | **4** | 4 | Tickets `T0161`, `T0162`, `T0163`, `T0164` (~520 words each). |
| Long Ticket Estimated Tokens | **734** | 734 | Each over-length ticket estimated at exactly 734 tokens (> 400 limit). |
| Eligible Tickets Dispatched to Classifier | **156** | 156 | Exactly 156 tickets proceed to PII redaction and model triage. |

---

### 5.3 Personal Identifiable Information (PII) Redaction (`src/part_c/pii_redaction.py`)
To prevent customer data exfiltration, sensitive personal identifiers are masked using regex substitution in strict sequential order prior to constructing prompts:
1. **Email Pattern**: `r"[\w.+-]+@[\w-]+\.[\w.]+"` $\rightarrow$ `[EMAIL]`
2. **Phone Pattern**: `r"\b[6-9]\d{9}\b"` $\rightarrow$ `[PHONE]`
3. **Order ID Pattern**: `r"ORD-\d{6}"` $\rightarrow$ `[ORDER_ID]`

#### PII Substitution Audit
| Identifier Type | Regex Pattern | Replacement Token | Substitutions Count | Reference |
|---|---|---|---:|---:|
| Email Address | `[\w.+-]+@[\w-]+\.[\w.]+` | `[EMAIL]` | **55** | 55 |
| Indian Phone Number | `\b[6-9]\d{9}\b` | `[PHONE]` | **91** | 91 |
| Order Identifier | `ORD-\d{6}` | `[ORDER_ID]` | **132** | 132 |
| **Total PII Masked** | — | — | **278** | **278** |
| Residual PII After Masking | — | — | **0** | **0** |

> [!IMPORTANT]
> **Zero-Leakage Assertion**: Re-applying all three regular expressions across the entire redacted corpus yields **0 residual matches**. In accordance with capstone requirements, the user message sent to `mock_llm.chat()` is strictly the redacted ticket text and nothing else.

---

### 5.4 Prompt Engineering: Zero-Shot vs Few-Shot (`src/part_c/prompts.py`)
Two system prompts were designed to evaluate classification capability, robustness, and token overhead:
- **Zero-Shot Prompt**: Contains the mandatory keyword `CLASSIFY`, lists the four allowed categories (`billing`, `delivery`, `technical`, `fraud`), and mandates JSON output of the form `{"category": "...", "reason": "..."}` without containing the substring `EXAMPLES:`.
- **Few-Shot Prompt**: Contains the mandatory keyword `CLASSIFY`, lists the four allowed categories, mandates identical JSON output, and includes a dedicated section header `EXAMPLES:` followed by four distinct labeled exemplars covering each category (including fraud).

---

### 5.5 Resilient Invocation & Exponential Backoff (`src/part_c/retry_handler.py`)
The underlying chat completion service simulates transient failures (`mock_llm.TransientLLMError: 503 server busy`) on every 9th call (`_state["calls"] % 9 == 0`).

To ensure zero dropped requests, all model interactions are wrapped in an autonomous retry handler:
- **Maximum Retries**: 3 retries per ticket ($n \in \{0, 1, 2\}$).
- **Exponential Backoff with Jitter**:
  $$\text{wait}(n) = \text{BASE\_DELAY} \times 2^n \times \text{Uniform}(0.9, 1.15), \quad \text{where } \text{BASE\_DELAY} = 0.05\text{s}$$
- **Jitter Range**: For retry attempt $n=0$, wait intervals lie strictly between **0.0450s and 0.0575s**.
- **Audit Logging**: Each retry emits a standardized audit line:
  `[RETRY] Ticket=<id> | retry=<n> | wait=<wait>s`

#### Retry Resilience Checkpoints
| Metric | Measured Value | Reference Checkpoint | Status |
|---|---:|---:|:---:|
| Full Run Eligible Tickets | 156 | 156 | Matched |
| Simulated Transient Failures Encountered | **19** | 19 | Matched |
| Failures Recovered on Retry 0 | **19** | 19 | Matched |
| Minimum Retry Wait Recorded | **0.0459s** | $\ge 0.0450\text{s}$ | Within Bounds |
| Maximum Retry Wait Recorded | **0.0570s** | $\le 0.0575\text{s}$ | Within Bounds |
| Unrecovered / Permanent Errors | **0** | 0 | 100% Resilience |

---

### 5.6 Output Validation & Keyword Fallback (`src/part_c/output_validation.py`)
Model responses are validated through a robust two-tiered parser that never guesses:
1. **JSON Validation (`classified_json`)**: The response is parsed with `json.loads()`. If parsing succeeds and `category` is within the four allowed values (`billing`, `delivery`, `technical`, `fraud`), the ticket receives status `classified_json`. Valid JSON containing an unauthorized category (such as the planted `{"category": "shipping", "reason": "parcel movement"}`) is immediately rejected and marked `flagged_invalid_output`.
2. **Word Extraction Fallback (`classified_keyword`)**: For non-JSON responses (such as the planted chatty response `The customer appears to have a <category>-related problem.`), tokens are extracted using `re.findall(r"[a-z]+", content.lower())`. This correctly splits hyphenated tokens like `billing-related` into `billing` and `related`. If exactly one allowed category word is present, it is accepted with status `classified_keyword`. If zero or multiple category words appear, it is assigned `flagged_invalid_output`.
3. **Human Review Escaping (`flagged_invalid_output`)**: Malformed outputs and out-of-spec categories are never guessed; they are flagged for human inspection.

#### Output Validation Status Breakdown (156 Eligible Tickets)
| Validation Status | Count | Reference | Operational Interpretation |
|---|---:|---:|---|
| `classified_json` | **135** | 135 | Well-formed JSON containing an authorized category. |
| `classified_keyword` | **12** | 12 | Non-JSON chatty replies safely rescued via regex word extraction. |
| `flagged_invalid_output` | **9** | 9 | Out-of-vocabulary categories (`shipping`) routed to human agents. |
| **Total Classified Tickets** | **147** | **147** | Successfully triaged tickets entering business queues ($135 + 12$). |

---

### 5.7 Zero-Shot vs Few-Shot Performance Comparison (`src/part_c/pipeline.py`)
Both prompt architectures were evaluated over a complete run across all 156 eligible tickets with no artificial token budget constraint (`TOKEN_BUDGET = 100,000`):

#### Quantitative Model Comparison
| Metric | Zero-Shot Prompt | Few-Shot Prompt | Delta / Impact |
|---|---:|---:|---|
| Tickets Ingested | 164 | 164 | Identical corpus |
| Guardrail Intercepts (`skipped_empty` + `flagged_long`) | 8 | 8 | Zero model cost |
| Model Invocations | 156 | 156 | 19 retries each |
| Output Rule Failures (`flagged_invalid_output`) | 9 | 9 | Deterministic misbehavior |
| Total Successfully Classified | **147** | **147** | Identical classified set |
| Correct Classifications vs `true_category` | 126 | **147** | +21 correct |
| **Classification Accuracy** | **85.71%** (126 / 147) | **100.00%** (147 / 147) | **+14.29%** |
| Total Token Consumption | **15,484** | **32,800** | +17,316 (+111.8%) |
| Average Tokens per Classified Ticket | **105.33** | **223.13** | +117.80 tokens/ticket |

#### Zero-Shot Confusion Matrix (147 Classified Tickets)
```
                  PREDICTED CATEGORY
TRUE CATEGORY   billing   delivery   technical   fraud    Total
billing            38         0          0         0       38
delivery            0        36          0         0       36
technical          10         0         29         0       39
fraud              11         0          0        23       34
Total              59        36         29        23      147
```

#### Detailed Zero-Shot Misclassifications
- **Fraud $\rightarrow$ Billing (11 errors)**: Tickets containing unauthorized charges or card hack reports were misclassified as standard billing issues because the zero-shot prompt lacked disambiguation exemplars, defaulting to billing keyword priority (`charged`, `deducted`, `payment`).
- **Technical $\rightarrow$ Billing (10 errors)**: Tickets reporting cart freeze or coupon application errors with pending order tags were erroneously classified as billing disputes.
- **Few-Shot Resolution**: Incorporating explicit `EXAMPLES:` with dedicated fraud and technical exemplars inverted the evaluation priority, eliminating all 21 misclassifications and achieving **100% classification accuracy**.

#### When Are the Extra Few-Shot Tokens Worth Paying For?
The few-shot prompt consumes an additional 117.80 tokens per classified ticket (223.13 vs 105.33 tokens), representing an approximate 2.12x cost increase. In a commercial quick-commerce environment handling thousands of daily tickets, this extra cost is overwhelmingly justified. Under zero-shot classification, 32.4% of actual fraud tickets (11 of 34) are misrouted to the regular billing support queue, delaying account security intervention and directly risking chargebacks, regulatory penalties, and customer churn. Investing approximately 118 extra tokens per ticket eliminates 100% of these high-risk routing errors, dramatically reducing human remediation overhead and financial liability.

---

### 5.8 Token Budget Experiment (`outputs/triage_budget_demo.csv`)
To protect against runaway API consumption, a hard cumulative token ceiling was evaluated:
- **Enforcement Rule**: Before each model invocation, if the running cumulative token total ($\sum \text{usage.total\_tokens}$) is $\ge \text{TOKEN\_BUDGET}$, the call is suppressed and the ticket receives status `budget_exhausted`.
- **Experimental Parameters**: Executed on the zero-shot pipeline with `TOKEN_BUDGET = 6,000`.
- **Output Artifact**: Persisted to `outputs/triage_budget_demo.csv`.

#### Budget Demo Results
| Status Category | Ticket Count | Action Taken |
|---|---:|---|
| `budget_exhausted` | **95** | Intercepted once cumulative tokens reached $\ge 6,000$; routed to `human_review`. |
| `classified_json` | 54 | Successfully processed within budget ceiling. |
| `classified_keyword` | 4 | Rescued keyword tickets processed within budget. |
| `flagged_invalid_output` | 3 | Invalid outputs occurring before budget exhaustion. |
| `skipped_empty` | 4 | Filtered pre-model by input guardrails (0 tokens). |
| `flagged_long` | 4 | Filtered pre-model by input guardrails (0 tokens). |
| **Total Processed** | **164** | **100% accounting of all customer tickets.** |

---

### 5.9 Customer Segment-Driven Routing & Final Production Results (`outputs/triage_results.csv`)
The production triage pipeline executes the **few-shot prompt** under an enterprise ceiling (`TOKEN_BUDGET = 100,000`), integrating customer segment intelligence from Part B (`outputs/customer_segments.csv`):

#### 4-Tier Routing Logic
```
Rule 1: If status in ['skipped_empty', 'flagged_long', 'flagged_invalid_output', 'budget_exhausted']
        ──► route = "human_review", priority = "NORMAL"
Rule 2: Else if category == "fraud"
        ──► route = "security_team", priority = "HIGH"
Rule 3: Else if customer is_premium == True (from Part B segments)
        ──► route = "premium_desk", priority = "HIGH"
Rule 4: Else
        ──► route = "support_queue", priority = "NORMAL"
```

#### Final Production Triage Routing Distribution (`outputs/triage_results.csv`)
| Routing Destination | Priority | Ticket Count | Percentage | Operational Purpose |
|---|:---:|---:|---:|---|
| `support_queue` | `NORMAL` | **102** | 62.20% | Standard automated routing for routine delivery, billing, and technical tickets from standard customers. |
| `security_team` | `HIGH` | **34** | 20.73% | High-priority escalation for all detected fraud and unauthorized transaction tickets. |
| `human_review` | `NORMAL` | **17** | 10.37% | Manual review for guardrail violations (4 empty, 4 long) and model validation failures (9 invalid output). |
| `premium_desk` | `HIGH` | **11** | 6.71% | White-glove concierge support queue for high-value Premium tier customers (`is_premium == True`). |
| **Total Ingested** | — | **164** | **100.0%** | Complete corpus routed deterministically. |

#### Priority Breakdown
- **`NORMAL` Priority**: 119 tickets (102 standard support queue + 17 human review).
- **`HIGH` Priority**: 45 tickets (34 fraud security team + 11 premium customer desk).

#### Output File Schema
`outputs/triage_results.csv` contains exactly 164 rows and 8 standardized columns:
- `ticket_id`: Primary ticket identifier (`T0001` .. `T0164`).
- `customer_id`: Instamart customer reference (`C0001` .. `C1200`).
- `status`: Final triage disposition (`classified_json`, `classified_keyword`, `skipped_empty`, `flagged_long`, `flagged_invalid_output`).
- `category`: Extracted category (`billing`, `delivery`, `technical`, `fraud`), left blank for unclassified tickets.
- `segment_name`: Segment description mapped from Part B (`Premium Loyalists`, `Frequent Regulars`, etc.).
- `priority`: Escalation level (`HIGH` or `NORMAL`).
- `route`: Operational dispatch destination (`support_queue`, `security_team`, `premium_desk`, `human_review`).
- `tokens_used`: Actual tokens consumed for the ticket interaction.

---

### 5.10 Part C Automated Validation Suite
Executing `python scripts/validate_part_c.py` verifies all eight Part C architectural requirements and numerical checkpoints:

```
============================================================
RUNNING PART C VERIFICATION SUITE
============================================================
[CHECK 1] Guardrails verified: 4 empty tickets, 4 long tickets (each 734 tokens).
[CHECK 2] PII redaction verified: 55 emails, 91 phones, 132 orders (278 total, 0 remaining).
[CHECK 3] System prompts verified: Zero-shot & Few-shot contract compliant.
[CHECK 4] Retry resiliency verified: exactly 19 TransientLLMErrors recovered on retry 0 within [0.045, 0.0575]s.
[CHECK 5] Output validation counts verified: 135 JSON, 12 keyword, 9 invalid output (147 classified).
[CHECK 6] Classification accuracies verified: Zero-shot 85.7% (126/147, 11 fraud & 10 tech -> billing), Few-shot 100.0% (147/147).
[CHECK 7] Token budget demo verified: 95 tickets ended budget_exhausted under 6000 token ceiling.
[CHECK 8] Routing & priorities verified: human_review (NORMAL), security_team (HIGH), premium_desk (HIGH), support_queue (NORMAL).
============================================================
ALL PART C CHECKPOINTS PASSED
============================================================
```

---

## 6. Part D: Policy RAG Assistant and Refund Agent

### 6.1 Architecture & Workflow Overview
Customer support in quick-commerce operations requires fast, grounded policy lookups and autonomous, audit-trailed refund resolution. Part D delivers two integrated AI capabilities:
1. **Policy Question-Answering Assistant (RAG)**: A retrieval-augmented generation engine that chunks operating policy documents into deterministic 40-word sliding windows with 10-word overlap, builds a TF-IDF vector space, retrieves relevant context chunks via cosine similarity, and synthesizes strictly grounded answers using `mock_llm.chat()` under the `POLICY ASSISTANT` harness.
2. **Autonomous ReAct Refund Agent**: A tool-using reasoning agent that interacts with the analytical SQLite database from Part A (`data/instamart.db`), policy retrieval, and a deterministic refund calculator to decide refund eligibility, record ledger entries (`outputs/refund_ledger.jsonl`), enforce multi-tier safety limits, and persist execution snapshots to `state/` for resumable crash recovery.

```
Incoming Customer Inquiry / Refund Request
                    │
                    ▼
       ┌────────────────────────┐
       │ Inbound Request Router │
       └───────────┬────────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
[Policy RAG Assistant]  [Autonomous ReAct Refund Agent]
  - 15 Chunks (TF-IDF)    - Tools: get_order, search_policy,
  - Cosine Similarity              calculate_refund, issue_refund,
  - Grounded Synthesis             escalate_to_human
  - Fallback Handling     - Safety: Timeout, Token, Max Iterations
                          - Crash Recovery: state/*.json Checkpoints
                          - Ledger: outputs/refund_ledger.jsonl
```

---

### 6.2 Policy Document Ingestion & Chunking (`src/part_d/chunking.py`)
Five official company policy documents are ingested from `data/policies/`:
1. `refund_policy.txt`: Rules governing completed order delivery latency, thresholds, approval tiers, and claims windows.
2. `delivery_policy.txt`: Promised SLA delivery times, fee thresholds, rain extensions, and OTP requirements.
3. `payment_policy.txt`: Payment channels (UPI, cards, COD limits), failed transaction turnaround, and wallet terms.
4. `privacy_policy.txt`: Customer data handling, address retention periods (180 days), and deletion timelines.
5. `escalation_policy.txt`: Fraud ticket priority (2-hour SLA), premium tier escalation, and agent handoff ceilings.

#### Deterministic Sliding-Window Chunking Algorithm
- **Configuration**: `CHUNK_SIZE = 40` words, `CHUNK_OVERLAP = 10` words (stride $= 30$ words).
- **Algorithm**: The document text is split on whitespace into words. Chunk $0$ spans words $0$ to $39$; Chunk $1$ starts at word $30$ and spans to word $69$; subsequent chunks advance by $30$ words. The process terminates immediately after the first chunk whose boundary reaches or exceeds the document word count. Each chunk records its originating filename.

#### Policy Chunk Distribution Checkpoints
| Policy Document | Total Words | Overlapping Chunks (`overlap=10`) | Zero-Overlap Chunks (`overlap=0`) | Status |
|---|---:|---:|---:|:---:|
| `refund_policy.txt` | 119 | **4** | 3 | Matched |
| `delivery_policy.txt` | 100 | **3** | 3 | Matched |
| `escalation_policy.txt` | 78 | **3** | 2 | Matched |
| `payment_policy.txt` | 86 | **3** | 3 | Matched |
| `privacy_policy.txt` | 68 | **2** | 2 | Matched |
| **Total Corpus** | **451** | **15** | **13** | **Exact Parity** |

---

### 6.3 TF-IDF Vectorization & Cosine Similarity Retrieval (`src/part_d/vector_store.py`)
Scikit-learn's `TfidfVectorizer()` is fitted across all 15 chunk texts to construct a sparse document-term matrix. Incoming queries are transformed into the same vector space, and chunks are ranked by cosine similarity:
$$\text{Cosine Similarity}(q, d) = \frac{q \cdot d}{\|q\|_2 \|d\|_2}$$

#### Understanding TF-IDF
- **Term Frequency (TF)** measures how frequently a given word appears in a specific chunk relative to the chunk length, reflecting local word significance.
- **Inverse Document Frequency (IDF)** measures how rare a word is across the entire corpus by taking the logarithm of the inverse chunk frequency, heavily weighting discriminative keywords while downweighting ubiquitous terms.
- **Core Limitation of TF-IDF**: TF-IDF relies strictly on surface-level lexical matching and possesses no semantic awareness. Queries expressing identical business concepts with alternative synonyms (e.g., *"spoiled milk"* vs *"expired dairy"*, or *"rider uncontactable"* vs *"driver not answering"*) yield near-zero similarity scores if exact term overlap is absent.

#### Retrieval Verification
For the reference query `"refund policy late delivery"`, the top three ranked chunks all originate from `refund_policy.txt`:
1. `refund_policy.txt` (Score: **0.2719**): `"...order is delivered in more than 45 minutes, the customer receives a full refund..."`
2. `refund_policy.txt` (Score: **0.2514**): `"Refund Policy. A refund can be requested only for orders whose status is completed..."`
3. `refund_policy.txt` (Score: **0.2454**): `"...late-delivery refund is paid. Refunds up to INR 1000 are approved automatically..."`

---

### 6.4 Grounded Policy RAG Assistant (`src/part_d/policy_assistant.py`)
`answer(question)` retrieves the top $k=3$ chunks, formats them into a strictly structured user payload, and prompts `mock_llm.chat()` under the `POLICY ASSISTANT` system marker:

#### Prompt Contract
```text
SYSTEM:
You are an automated POLICY ASSISTANT. Answer the customer's question strictly using the provided policy context. Do not invent or extrapolate policies. If the answer cannot be found in the context, state that you could not find that information.

USER:
CONTEXT:
[refund_policy.txt] ...
[delivery_policy.txt] ...
[payment_policy.txt] ...

QUESTION:
What delivery time makes a completed order eligible for a full refund?
```

#### Evaluation on 14 Questions (`data/raw/policy_eval.csv`)
| Evaluation Category | Test Questions | Passed | Success Rate | Checkpoint Reference |
|---|---:|---:|---:|:---:|
| Expected Document in Top 3 Sources | 12 | **12** | **100.0%** | 12/12 |
| Expected Fact Grounded in Answer (Case-Sensitive) | 12 | **12** | **100.0%** | 12/12 |
| Unanswerable Questions Rejected with Fallback | 2 | **2** | **100.0%** | 2/2 |

Both unanswerable questions (company mascot colour, cricket world cup winner) were rejected with the exact fallback:
`"I could not find that information in the policy documents."`

---

### 6.5 Zero-Overlap Comparison Experiment (`src/part_d/retrieval.py`)
To quantify the benefit of the 10-word sliding window overlap, the entire grounding evaluation was repeated using non-overlapping chunks (`chunk_size=40, chunk_overlap=0`):

#### Overlap vs Zero-Overlap Retrieval & Grounding Performance
| Metric | Overlapping Chunks (`overlap=10`) | Zero-Overlap Chunks (`overlap=0`) | Impact of Overlap |
|---|---:|---:|---|
| Total Chunks | 15 | 13 | +2 chunks (+15.4%) |
| Answerable Document in Top 3 | 12 / 12 (100.0%) | 12 / 12 (100.0%) | Identical top document retrieval |
| Answerable Fact Grounded | **12 / 12 (100.0%)** | **10 / 12 (83.33%)** | **-2 failed questions** |
| Unanswerable Correctly Rejected | 2 / 2 (100.0%) | 2 / 2 (100.0%) | Identical fallback handling |

#### Why Zero-Overlap Breaks Policy Answers
When `overlap=0`, natural sentences are arbitrarily truncated at fixed word boundaries:
1. **Question 1**: *"What delivery time makes a completed order eligible for a full refund?"*
   - In `refund_policy.txt`, the clause `"...delivered in more than 45 minutes, the customer receives a full refund of the order amount."` is split across chunk 0 and chunk 1. Chunk 1 retrieves the fragment `"receives a full refund of the order amount."`, completely omitting the prerequisite clause `"more than 45 minutes"`.
2. **Question 8**: *"How soon is money returned when a payment fails but is deducted?"*
   - In `payment_policy.txt`, the condition and resolution `"If a payment fails but money is deducted, the amount is returned automatically within 48 hours."` has its terminal phrase split, causing the model to retrieve an incomplete sentence missing `"48 hours"`.

Sliding window overlap ensures boundary continuity, preserving semantic integrity across adjacent windows.

---

### 6.6 Tool Specifications & Dispatch Registry (`src/part_d/tools.py`, `src/part_d/tool_registry.py`)
Five tools were implemented to empower the autonomous refund agent:

| Tool Name | Description | JSON Schema Parameters | Return Structure |
|---|---|---|---|
| `get_order` | Query analytical SQLite DB (`data/instamart.db`) for order status, amount, and delivery minutes. | `order_id` (string) | `{"order_id", "status", "order_amount", "delivery_minutes"}` or `{"error": "order not found"}` |
| `search_policy` | Retrieve grounded policy clauses via TF-IDF cosine similarity. | `query` (string) | `{"results": [...]}` |
| `calculate_refund` | Compute policy-governed refund in INR based on order amount and delivery duration. | `order_amount` (number), `delivery_minutes` (integer) | `{"refund_inr": <amount>}` |
| `issue_refund` | Record and disburse approved refund ($\le \text{INR } 1000$) with transaction-level idempotency to `outputs/refund_ledger.jsonl`. | `order_id` (string), `refund_inr` (number), `idempotency_key` (string, optional) | `{"refund_inr": <amount>, "status": "issued", "idempotency_key": ...}` or `{"error": ...}` |
| `escalate_to_human` | Escalate high-value refund ($> \text{INR } 1000$) to human lead and record in ledger. | `order_id` (string), `refund_inr` (number), `idempotency_key` (string, optional) | `{"refund_inr": <amount>, "status": "escalated", "idempotency_key": ...}` or `{"error": ...}` |

#### Refund Decision Business Rules
1. **Completion Prerequisite**: Only orders with `status == "completed"` are eligible for refunds. Cancelled orders were never billed ($\text{refund} = \text{INR } 0.00$).
2. **Delivery Latency Threshold**: Completed orders delivered in $\le 45$ minutes are on-time ($\text{refund} = \text{INR } 0.00$).
3. **Full Refund SLA**: Completed orders delivered in $> 45$ minutes receive a **100% full refund** of `order_amount`. Partial refunds are prohibited.
4. **Approval Threshold**: Eligible refunds up to $\text{INR } 1000$ are automatically approved via `issue_refund`. Refunds strictly exceeding $\text{INR } 1000$ mandate human approval via `escalate_to_human` and must never be issued automatically.

---

### 6.7 Financial Safety & Transaction-Level Idempotency (`src/part_d/tools.py`)
To prevent unauthorized payouts and duplicate disbursements, financial safety is enforced at the tool boundary:
1. **Tool-Enforced INR 1,000 Refund Ceiling**:
   - `issue_refund()` validates amounts directly: values between INR 0.00 and INR 1,000.00 are accepted.
   - Any requested amount strictly exceeding INR 1,000.00 (e.g., INR 1,000.01 or INR 1,999.00) is rejected immediately with an explicit error response (`"error": "Refund amount INR ... exceeds maximum auto-approval limit of INR 1000.00."`).
   - Rejected refunds are **never** appended to `outputs/refund_ledger.jsonl`.
   - Invalid, negative, NaN, and infinite values are strictly rejected without ledger mutation.
   - `escalate_to_human()` validates escalation amounts and explicitly supports high-value claims ($> \text{INR } 1,000$).
2. **Transaction-Level Idempotency Protection**:
   - Every refund execution establishes a stable transaction key (`idempotency_key`). In ReAct agent execution, this key defaults to `f"{run_id}:{order_id}"`, tying the transaction to the specific agent execution.
   - Repeated requests bearing the same idempotency key are rejected with `status = "duplicate"` and do not append a duplicate payout record to the ledger.
   - Separate legitimate transactions can reuse an order ID across distinct runs or simulations without being blocked globally, because deduplication is transaction-keyed rather than order-keyed.
   - The ledger explicitly records `idempotency_key` for auditability.
3. **Architecture Boundaries: Implemented Guarantees vs. Production Limitations**:
   - **Implemented & Verified**: In-process thread-safe locking (`_LEDGER_LOCK`), key-based duplicate verification against the local ledger file, duplicate rejection, and tool-level amount validation.
   - **Production Limitations (Not Implemented)**: The local JSONL ledger does **not** provide ACID transactional guarantees (no multi-phase commits, write-ahead logs, or atomic rollbacks). File-locking is process-local and single-node; enterprise deployments require an ACID-compliant transactional SQL database (e.g. PostgreSQL) with unique database constraints on idempotency keys and row-level locking (`SELECT ... FOR UPDATE`).

---

### 6.8 ReAct Agent Loop & Scenario Execution (`src/part_d/agent.py`)
`run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10)` executes an autonomous reasoning and action loop. All 7 rows from `data/raw/agent_scenarios.csv` were executed, logging audit trails to `outputs/agent_runs.jsonl`:

#### Seven Agent Scenario Execution Results
| Scenario | Inbound Message / Order ID | Status | Refund Amount (INR) | Model Calls | Tokens Used | Tools Executed Sequence |
|---|---|---|---:|---:|---:|---|
| **S1** | `ORD-100048` (Late, small order) | `REFUND` | **INR 128.25** | **5** | 1,147 | `get_order` $\rightarrow$ `search_policy` $\rightarrow$ `calculate_refund` $\rightarrow$ `issue_refund` |
| **S2** | `ORD-100067` (Late, small order) | `REFUND` | **INR 135.00** | **5** | 1,147 | `get_order` $\rightarrow$ `search_policy` $\rightarrow$ `calculate_refund` $\rightarrow$ `issue_refund` |
| **S3** | `ORD-100145` (Late, high-value order) | `ESCALATE` | **INR 1,999.00** | **5** | 1,134 | `get_order` $\rightarrow$ `search_policy` $\rightarrow$ `calculate_refund` $\rightarrow$ `escalate_to_human` |
| **S4** | `ORD-100001` (On-time, delivered 18m) | `DENIED` | **INR 0.00** | **4** | 805 | `get_order` $\rightarrow$ `search_policy` $\rightarrow$ `calculate_refund` |
| **S5** | `ORD-100025` (Cancelled order) | `DENIED` | **INR 0.00** | **3** | 487 | `get_order` $\rightarrow$ `search_policy` |
| **S6** | `ORD-999999` (Non-existent order) | `NOT_FOUND` | **INR 0.00** | **2** | 172 | `get_order` |
| **S7** | `ORD-100048` with `[LOOP]` | `FALLBACK_MAX_ITERATIONS` | **INR 0.00** | **6** | 3,300 | `search_policy` (repeated 6 times) |

> [!NOTE]
> Every scenario refund amount matches `expected_refund_inr` within INR 0.01. Scenario S1 completed in exactly **5 model calls**, and scenario S7 executed all 6 iterations before gracefully stopping with `FALLBACK_MAX_ITERATIONS` under the default 6000 token ceiling (consuming 3,300 tokens).

---

### 6.9 Safety Limits & Graceful Fallback Demonstrations
Three independent safety boundaries protect system resources:
1. **Iteration Limit**: After 6 calls without terminal resolution, S7 terminates with `FALLBACK_MAX_ITERATIONS`.
2. **Token Ceiling Guard**: Evaluated after every model call. When `tokens_used > token_limit`, the agent halts immediately without executing the requested tool.
   - **Demonstration**: Scenario S1 executed with `token_limit = 200` halted after Call 1 (consuming 230 prompt/completion tokens) with status **`FALLBACK_TOKEN_LIMIT`**.
3. **Wall-Clock Timeout**: Evaluated at the top of every iteration using `time.monotonic()`. When `elapsed >= time_limit_seconds`, execution halts before model invocation.
   - **Demonstration**: Scenario S1 executed with `time_limit_seconds = 0` halted immediately before Call 1 with status **`FALLBACK_TIMEOUT`**.

---

### 6.10 Resumable Execution & Crash Recovery Demonstration (`src/part_d/checkpoint.py`, `src/part_d/resume.py`)
After each tool observation is appended, a state snapshot is saved to `state/<run_id>.json` recording `run_id`, `scenario_id`, `messages`, `tokens_used`, `llm_calls`, `step_count`, and `tools_executed`.

#### Required Crash Demonstration on Scenario S1
- **Phase 1 (Initial Run & Crash)**:
  - Tool 1 executed: `get_order` (order details loaded, checkpoint saved).
  - Tool 2 executed: `search_policy` (policy retrieved, checkpoint saved).
  - Simulated crash triggered: `SimulatedAgentCrash: Simulated crash after executing tool 2: search_policy (run_id: crash_recovery_demo_s1)`.
- **Phase 2 (Resumed Execution via `resume_agent`)**:
  - Checkpoint loaded from `state/crash_recovery_demo_s1.json`.
  - Context restored with conversation history and previous tool results.
  - Tool 3 executed: `calculate_refund` (amount INR 128.25 calculated).
  - Tool 4 executed: `issue_refund` (recorded in ledger).
  - Final answer returned: `"REFUND: INR 128.25 issued."` with status **`REFUND`**.

#### Exact Sequence of Tools Executed Across Crash and Resume
```text
1. get_order          (Executed in initial run before crash)
2. search_policy      (Executed in initial run before crash; snapshot saved)
[CRASH SIMULATED]
[RESUME_AGENT CALLED]
3. calculate_refund   (Executed in resumed run; restored from checkpoint)
4. issue_refund       (Executed in resumed run; ledger updated once)
```
- **Total Combined Tool Executions**: Exactly **4** tools.
- **Zero Duplication**: No tool was re-executed.
- **Ledger Integrity**: `outputs/refund_ledger.jsonl` contains exactly **one** refund issuance for `ORD-100048` from the crash demonstration run.

---

### 6.11 ReAct versus Plan-and-Execute Architecture Comparison

> In this quick-commerce refund workflow, the **ReAct (Reasoning + Acting)** architecture is fundamentally superior to a **Plan-and-Execute** design.
>
> **Model Calls & Token Efficiency**: Plan-and-execute prompts the LLM once upfront to generate a full sequential plan (`get_order` $\rightarrow$ `search_policy` $\rightarrow$ `calculate_refund` $\rightarrow$ `issue_refund`), executing tools deterministically and replanning only on failure. For straightforward happy paths like S1 (5 calls, 1,147 tokens), plan-and-execute would reduce LLM calls to 1 or 2. However, in real-world triage, inquiries frequently branch or fail early: S6 (`NOT_FOUND`) terminates in 2 calls (172 tokens), and S5 (cancelled order) terminates in 3 calls (487 tokens). Plan-and-execute would waste tokens pre-generating invalid steps (e.g., attempting to calculate and issue refunds for non-existent orders) before encountering execution exceptions.
>
> **Handling Unexpected Observations**: ReAct naturally inspects runtime observations dynamically. When `get_order` returns `{"error": "order not found"}`, ReAct immediately halts and responds with `NOT_FOUND`. A plan-and-execute agent requires complex error-handling and replanning prompts when reality diverges from static assumptions.
>
> **Architectural Recommendation**: For financial transaction workflows where data integrity, idempotency, and early termination on invalid orders are paramount, ReAct provides adaptive grounding, minimal unnecessary tool invocations, and strict step-by-step verification.

---

### 6.12 Part D Automated Validation Status
Executing `python scripts/validate_part_d.py` validates all Part D functional requirements:

```text
============================================================
RUNNING PART D COMPREHENSIVE VALIDATION SUITE
============================================================
  [PASS] Task 1: 5 policy files loaded, exactly 15 chunks (refund 4, delivery 3, escalation 3, payment 3, privacy 2).
  [PASS] Task 2: TF-IDF retrieval top 3 chunks for 'refund policy late delivery' all from refund_policy.txt with calculated cosine scores.
  [PASS] Tasks 3 & 4: Grounding evaluation passed with 12/12 doc retrieved, 12/12 fact grounded, 2/2 unanswerable fallback.
  [PASS] Task 4 Zero-Overlap: Successfully reproduced failure in zero-overlap experiment (2 questions failed due to chunk boundary splitting).
  [PASS] Task 5: Tool specs, dispatch map, DB lookups, and refund calculation rules verified.
  [PASS] Financial Safety & Idempotency: All 10 direct safety tests verified (INR 1000 cap, rejected amounts, NaN/Inf, duplicate keys, separate transactions, crash idempotency).
  [PASS] Task 6: All seven scenarios matched expected outcomes, S1 took 5 calls, S7 reached FALLBACK_MAX_ITERATIONS.
  [PASS] Task 7: Token limit demo (FALLBACK_TOKEN_LIMIT) and Timeout demo (FALLBACK_TIMEOUT) verified.
  [PASS] Task 8: Crash and resume demonstration verified (exact 4 tool sequence: get_order, search_policy, calculate_refund, issue_refund; status REFUND INR 128.25).
  [PASS] Artifacts: Outputs ledger validated with 4 records (both issue_refund and escalate_to_human present).
============================================================
ALL PART D CHECKPOINTS PASSED
============================================================
```

---

## 7. Limitations and Next Steps

### 7.1 Current System Limitations
1. **Synthetic Data Characteristics**: The data foundation uses synthetic, deterministically generated datasets. In real-world production systems, real-time streaming changes, partial payment retries, and network dropouts introduce non-deterministic anomalies.
2. **Fixed Category Imputation**: Imputation of missing delivery times for completed orders relies on static category medians rather than multi-variate modeling (e.g., city, weather conditions, hour of day).
3. **Single Analytical Window**: Transaction history spans exactly 183 days (April 1 to September 30, 2025). Seasonality across festival periods (Diwali, New Year) is not captured.
4. **Post-Hoc Whole-Window Premium Labeling**: The Premium flag originates from the whole-window RFM analysis (spanning through September 30), which overlaps with the September lapse outcome window; therefore, the prioritized retention list is illustrative and not a leakage-free evaluation.
5. **PII Masking Scope**: Current regex masking targets standard email structures, 10-digit Indian mobile numbers, and Swiggy order patterns. Unstructured personal identifiers (such as street addresses, customer names, or UPI IDs) require named entity recognition (NER) models for complete redaction in open production environments.
6. **Token Budgeting Dynamics**: The token budget manager operates at batch-level cut-offs; production implementations would benefit from customer-level rate-limiting and dynamic prompt token compression.
7. **TF-IDF Lexical Limitations**: The policy retrieval engine is based on term-frequency inverted indexes. While effective for exact keyword matches, it cannot capture deep semantic paraphrasing; integrating dense embedding vector search (e.g. bi-encoders) would improve domain generalization.

8. **Local JSONL Ledger & Concurrency Boundary**: The refund ledger is an append-only JSON Lines file protected by in-memory thread locking (`_LEDGER_LOCK`). It does not provide ACID transactional guarantees (no write-ahead logs, two-phase commit, or multi-node atomic rollbacks). Production deployments require an ACID-compliant SQL database (e.g., PostgreSQL) with unique constraints on transaction keys, distributed locks, and atomic `SELECT ... FOR UPDATE` isolation.

### 7.2 Next Steps for Production Deployment
1. **Hybrid Dense-Sparse Retrieval**: Combine TF-IDF sparse matching with neural dense embeddings (BM25 + ColBERT/dense vector search) and cross-encoder re-ranking for enterprise policy retrieval.
2. **Distributed Checkpointing**: Migrate state checkpoints from local disk JSON files to a distributed cache (e.g., Redis or DynamoDB) with distributed locking for multi-worker agent scalability.
3. **Automated End-to-End Orchestration**: Package the entire four-part suite into a containerized service with unified REST endpoints, telemetry logging, and continuous evaluation dashboards.
