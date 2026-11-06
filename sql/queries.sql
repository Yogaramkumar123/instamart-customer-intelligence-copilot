-- ============================================================================
-- Instamart Customer Intelligence & Support Copilot
-- Part A: Seven SQL Queries
-- Database: data/instamart.db
-- ============================================================================

-- [Q1] WHERE and HAVING:
-- Completed orders with order_amount > 500, grouped by customer,
-- retaining customers whose qualifying order total exceeds INR 8,000.
-- Expected: 30 customers.
SELECT
    customer_id,
    ROUND(SUM(order_amount), 2) AS total_qualifying_spend,
    COUNT(order_id) AS qualifying_order_count
FROM orders
WHERE status = 'completed'
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_qualifying_spend DESC;

-- [Q2] Three-table JOIN:
-- Revenue grouped by city and product category.
-- Expected: 30 rows; Grand Total = INR 2,998,668.80; Largest cell = Bengaluru / Electronics (INR 233,909.45).
SELECT
    c.city,
    p.category,
    ROUND(SUM(o.order_amount), 2) AS revenue,
    COUNT(o.order_id) AS order_count
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city ASC, p.category ASC;

-- [Q3] Scalar Subquery:
-- Products priced above average catalogue price.
-- Expected: 11 products.
SELECT
    product_id,
    product_name,
    category,
    price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC, product_id ASC;

-- [Q4] LEFT JOIN:
-- Customers without a completed order.
-- Expected: 25 customers.
SELECT
    c.customer_id,
    c.city,
    c.signup_date
FROM customers c
LEFT JOIN orders o
    ON c.customer_id = o.customer_id
   AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id ASC;

-- [Q5] Common Table Expression (CTE):
-- Calculate completed spend per customer and average customer spend.
-- Return customers whose spend exceeds the average.
-- Expected: 550 customers; Average Spend = INR 2,552.06.
WITH customer_completed_spend AS (
    SELECT
        customer_id,
        SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
benchmark_spend AS (
    SELECT
        AVG(total_spend) AS avg_customer_spend
    FROM customer_completed_spend
)
SELECT
    cs.customer_id,
    ROUND(cs.total_spend, 2) AS total_spend,
    ROUND(b.avg_customer_spend, 2) AS avg_spend
FROM customer_completed_spend cs
CROSS JOIN benchmark_spend b
WHERE cs.total_spend > b.avg_customer_spend
ORDER BY cs.total_spend DESC, cs.customer_id ASC;

-- [Q6] Ranking Window Functions:
-- For each customer with completed spend: total spend, RANK(), DENSE_RANK(), and ROW_NUMBER()
-- partitioned by city and ordered by total spend descending.
-- Expected top per city:
-- Bengaluru: C0082 (12,856.20)
-- Chennai:   C0930 (10,151.40)
-- Delhi:     C0291 (13,742.60)
-- Hyderabad: C0743 (12,730.05)
-- Mumbai:    C0693 (14,766.55)
-- Pune:      C0426 (10,377.10)
-- Exactly 8 rows have RANK() != ROW_NUMBER() due to ties in customer spend.
WITH customer_city_spend AS (
    SELECT
        c.customer_id,
        c.city,
        ROUND(SUM(o.order_amount), 2) AS total_spend
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT
    city,
    customer_id,
    total_spend,
    RANK() OVER (
        PARTITION BY city
        ORDER BY total_spend DESC
    ) AS spend_rank,
    DENSE_RANK() OVER (
        PARTITION BY city
        ORDER BY total_spend DESC
    ) AS spend_dense_rank,
    ROW_NUMBER() OVER (
        PARTITION BY city
        ORDER BY total_spend DESC
    ) AS spend_row_number
FROM customer_city_spend
ORDER BY city ASC, spend_rank ASC, customer_id ASC;

-- [Q7] Running Total:
-- Daily completed revenue and cumulative revenue using a window function.
-- Expected: 183 daily rows; Final cumulative revenue = INR 2,998,668.80.
WITH daily_revenue AS (
    SELECT
        SUBSTR(order_ts, 1, 10) AS order_date,
        ROUND(SUM(order_amount), 2) AS daily_rev
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT
    order_date,
    daily_rev,
    ROUND(SUM(daily_rev) OVER (ORDER BY order_date ASC), 2) AS cumulative_revenue
FROM daily_revenue
ORDER BY order_date ASC;
