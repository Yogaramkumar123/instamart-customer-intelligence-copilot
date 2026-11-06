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
│   └── part_b/
│       ├── __init__.py
│       ├── rfm_analysis.py     # Customer RFM feature calculation
│       ├── clustering.py       # K-Means clustering (K=2..8) and selection
│       ├── segment_profiling.py# Cluster profiling, naming & export
│       ├── pca_analysis.py     # 2D PCA projection & variance analysis
│       ├── lapse_features.py   # Leakage-free cohort & feature matrix
│       ├── preprocessing.py    # Stratified split & ColumnTransformer
│       ├── model_training.py   # Baseline, Logistic Regression & Random Forest tuning
│       ├── class_balance.py    # Training-only random oversampling experiment
│       ├── model_interpretation.py # Feature importance & coefficient readings
│       ├── retention_list.py   # Illustrative Premium retention ranking
│       └── validate.py         # Comprehensive Part B validation suite
│
├── scripts/
│   ├── run_part_a.py           # End-to-end master orchestration pipeline (Part A)
│   ├── validate_part_a.py      # Standalone verification runner (Part A)
│   ├── run_part_b.py           # End-to-end master orchestration pipeline (Part B)
│   └── validate_part_b.py      # Standalone verification runner (Part B)
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
│   ├── part_a/                 # Query outputs, pivot tables, audit logs
│   └── part_b/                 # K-Means comparisons and PCA summaries
│
└── state/                      # Placeholder for execution state checkpoints
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
4. **Post-Hoc Whole-Window Premium Labeling**: The Premium flag originates from the whole-window RFM analysis (spanning through September 30), which overlaps with the September lapse outcome window; therefore, the prioritized retention list is illustrative and not a leakage-free evaluation.

### 7.2 Next Steps for Integration
1. **Support Ticket Triage Routing (Part C)**: Feed `outputs/customer_segments.csv` into Part C to route inbound support tickets from Premium customers directly to the `premium_desk` queue with high priority.
2. **Database Integration for Agents (Part D)**: Part D's autonomous refund agent will connect directly to `data/instamart.db` to retrieve order records and verify delivery latency against refund policies.
3. **End-to-End Orchestration**: Develop an overall repository runner linking data foundation, customer intelligence, ticket triage, and autonomous policy agents into a single operational interface.
