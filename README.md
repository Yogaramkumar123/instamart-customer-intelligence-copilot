# Instamart Customer Intelligence & Support Copilot

An enterprise-grade, reproducible data engineering and intelligence pipeline designed for quick-commerce operations across six metro cities in India. This repository delivers **Part A: Data Foundation — Clean, Store, Query** of the capstone project, turning raw, noisy transaction exports into a trusted, relational analytical database and answering core business intelligence questions through SQL and pandas.

---

## 1. Overview

### 1.1 Project Architecture
Quick-commerce environments like Swiggy Instamart process tens of thousands of orders across distributed dark stores within minutes. Operational data exports frequently suffer from data quality issues: duplicate records, partial duplicate pending states, negative delivery durations, corrupt basket amounts, non-standard categorical spellings, and anomalous customer demographics.

The **Instamart Customer Intelligence & Support Copilot** provides an end-to-end analytical foundation structured into four progressive stages:
- **Part A: Data Foundation (Clean, Store, Query)** — Ingestion, multi-table auditing, deterministic sequential data cleaning, relational SQLite data warehouse modeling, and analytical querying via SQL and pandas.
- **Part B: Customer Segmentation & Lapse Prediction** *(Future Integration)* — RFM feature extraction, K-Means clustering, PCA dimensionality reduction, and leakage-free customer lapse prediction.
- **Part C: Support Ticket Triage with Guardrails** *(Future Integration)* — PII masking, token budgeting, transient error recovery, and LLM-powered ticket classification.
- **Part D: Policy RAG Assistant & Refund Agent** *(Future Integration)* — Vectorized chunk retrieval over operating policies and an autonomous ReAct refund agent.

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
│   └── part_a/
│       ├── __init__.py
│       ├── audit.py            # Dataset structure and anomaly profiling
│       ├── clean_customers.py  # Customer demographic cleansing & imputation
│       ├── clean_products.py   # Product catalog validation
│       ├── clean_orders.py     # Sequential order cleaning & deduplication
│       ├── database.py         # Relational SQLite database pipeline
│       ├── sql_analysis.py     # Seven analytical business queries runner
│       ├── pandas_analysis.py  # Pivot tables, descriptive stats & correlations
│       └── validate.py         # Automated checkpoint validation suite
│
├── scripts/
│   ├── run_part_a.py           # End-to-end master orchestration pipeline
│   └── validate_part_a.py      # Standalone verification runner
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
│   └── part_a/                 # Query outputs, pivot tables, audit logs
│
└── state/                      # Placeholder for execution state checkpoints
```

### 1.2 Technology Stack & Dependencies
- **Language**: Python 3.11+ (Python 3.12 compatible)
- **Data Processing**: pandas (>=2.0.0), numpy (>=1.24.0)
- **Machine Learning & Storage**: scikit-learn (>=1.3.0), joblib (>=1.3.0)
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
   *This executes the audit, customer cleaning, product validation, sequential order cleaning, SQLite database creation, all seven SQL queries, and the pandas exploratory analysis.*

5. **Execute standalone checkpoint validation**:
   ```bash
   python scripts/validate_part_a.py
   ```
   *Output ends with the exact string:* `ALL CHECKPOINTS PASSED`

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
3. **Latest Record Retention by Order ID**: Grouped by `order_id` and retained only the record with the latest timestamp (`.sort_values('order_ts').groupby('order_id', as_index=False).last()`). This removed **656** earlier pending records, eliminating all pending rows and retaining the final status. Row count reduced to exactly **12,309**.
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
Executing `python scripts/validate_part_a.py` executes 36 automated assertions across file existence, schema validity, demographic distributions, order integrity, revenue consistency, SQL query results, and statistical distributions.

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

> [!NOTE]
> *Structured Placeholder for Part B.*

### 4.1 Scope & Objectives
Part B builds upon the cleaned foundation established in Part A (`data/clean/orders.csv` and `data/clean/customers.csv`) to provide actionable customer intelligence for marketing and retention teams:
- **RFM Matrix Formulation**: Calculate Recency (days to 2025-10-01), Frequency (distinct completed orders), and Monetary value (total completed spend) for 1,175 active customers.
- **K-Means Clustering**: Standardize RFM features, evaluate cluster inertia and silhouette scores across $K \in [2, 8]$, identify the optimal elbow, and project onto 2D space using PCA.
- **Segment Profiling**: Label customer archetypes (including designating the highest monetary cluster as `is_premium`).
- **Lapse Definition & Modeling**: Establish a temporal cutoff at 2025-09-01. Construct pre-cutoff behavioral features without data leakage, evaluate Logistic Regression and Random Forest classifiers using 5-fold cross-validation, and perform class-balance experiments.
- **Artifacts to be Produced**:
  - `outputs/customer_segments.csv`
  - `outputs/lapse_model.joblib`
  - `outputs/retention_list.csv`

---

## 5. Part C: Support Ticket Triage with Guardrails

> [!NOTE]
> *Structured Placeholder for Part C.*

### 5.1 Scope & Objectives
Part C processes 164 customer support tickets from `data/raw/tickets.csv` through an offline classification pipeline backed by `mock_llm.py`:
- **Input Guardrails**: Filter empty tickets (`skipped_empty`) and tickets exceeding 400 estimated tokens (`flagged_long`).
- **PII Redaction**: Mask email addresses (`[EMAIL]`), phone numbers (`[PHONE]`), and order identifiers (`[ORDER_ID]`) before prompt assembly.
- **Resilient LLM Invocation**: Handle transient errors (`TransientLLMError`) using exponential backoff with jitter.
- **Output Validation**: Enforce strict category JSON parsing and fallback keyword extraction (`classified_json`, `classified_keyword`, `flagged_invalid_output`).
- **Budgeting & Intelligent Routing**: Implement token expenditure budgeting and route tickets based on customer segment flags from Part B (`security_team`, `premium_desk`, `support_queue`, `human_review`).
- **Artifacts to be Produced**:
  - `outputs/triage_budget_demo.csv`
  - `outputs/triage_results.csv`

---

## 6. Part D: Policy RAG Assistant and Refund Agent

> [!NOTE]
> *Structured Placeholder for Part D.*

### 6.1 Scope & Objectives
Part D delivers an automated support assistant and autonomous refund resolution agent:
- **Policy Retrieval-Augmented Generation (RAG)**: Chunk operational policy documents (40 words with 10-word overlap), vectorize chunks using TF-IDF, and retrieve grounded contexts via cosine similarity to answer customer queries.
- **ReAct Refund Agent**: Autonomous agent capable of calling tools (`get_order` from `data/instamart.db`, `search_policy`, `calculate_refund`, `issue_refund`, `escalate_to_human`).
- **Safety & Fallback Mechanisms**: Guard against infinite loops (`FALLBACK_MAX_ITERATIONS`), enforce token limits (`FALLBACK_TOKEN_LIMIT`), and wall-clock timeouts (`FALLBACK_TIMEOUT`).
- **Resumability**: State persistence across tool calls in `state/` allowing crash recovery without duplicate executions.
- **Artifacts to be Produced**:
  - `outputs/agent_runs.jsonl`
  - `outputs/refund_ledger.jsonl`

---

## 7. Limitations and Next Steps

### 7.1 Current System Limitations
1. **Synthetic Data Characteristics**: The data foundation uses synthetic, deterministically generated datasets. In real-world production systems, real-time streaming changes, partial payment retries, and network dropouts introduce non-deterministic anomalies.
2. **Fixed Category Imputation**: Imputation of missing delivery times for completed orders relies on static category medians rather than multi-variate modeling (e.g., city, weather conditions, hour of day).
3. **Single Analytical Window**: Transaction history spans exactly 183 days (April 1 to September 30, 2025). Seasonality across festival periods (Diwali, New Year) is not captured.

### 7.2 Next Steps for Integration
1. **Seamless Part B Ingestion**: Part B will directly consume `data/clean/orders.csv` and `data/clean/customers.csv`, ensuring strict leakage-free temporal feature engineering.
2. **Database Integration for Agents**: Part D's refund agent will connect directly to `data/instamart.db` to retrieve order records and verify completion status.
3. **End-to-End Orchestration**: Develop an overall repository runner to link data preparation, model training, ticket triage, and autonomous policy agents into a single operational interface.
