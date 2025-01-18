---
layout: post
title: Amazon BIE/DE Practice
date: 2025-01-17 14:30 -0800
description: Practice Amazon interview questions
author: khoa_pham
categories: [Coding Workspace, Interview Preps]
tags: [data engineering, interview, coding, sql, python]
pin: true
math: true
mermaid: true
toc: true
comments: true
---

# SQL

## Joins

### All Orders (Inner Joins)

**Q1:** Write a query to find all orders placed by customers who live in New York.
**Tables:**
- `Orders`: order_id, customer_id, order_date.
- `Customers`: customer_id, name, city.

```sql
SELECT o.order_id, o.customer_id, o.order_date, c.name
FROM Orders o INNER JOIN Customers c
ON o.customer_id = c.customer_id
WHERE c.city = 'New York';
```

> The INNER JOIN ensures only rows with matching customer_id in both tables are included.

### All Products (Left Joins)

**Q2:** List all products, including those that have not been sold, with their total sales quantities.
**Tables:**
- `Products`: product_id, product_name.
- `Sales`: sale_id, product_id, sale_amount.

```sql
SELECT p.product_id, p.product_name,
       COALESCE(SUM(s.sale_amount), 0) AS total_sale_amount
FROM Products p LEFT JOIN Sales s
ON p.product_id = s.product_id
GROUP BY p.product_id, p.product_name;
```

> The LEFT JOIN includes all products, even those without matching sales.
> COALESCE handles NULL values by replacing them with 0 for unsold products.

### All Suppliers & Products (Full Outer Join)
**Q3:** Combine data from Suppliers and Products, showing all products and suppliers, even if there are no matches.
**Tables:**
- `Suppliers`: supplier_id, supplier_name.
- `Products`: product_id, product_name, supplier_id.

```sql
SELECT 
FROM Suppliers s FULL OUTER JOIN Products p
ON s.supplier_id = p.supplier_id;
```

> The FULL OUTER JOIN ensures inclusion of all rows from both tables, with NULL where no matches exist.

### All Employees (Self Join)
**Q4:** List all employees and their managers.
**Tables:**
- `Employees`: employee_id, name, manager_id.

```sql
SELECT e.name AS employee, m.name AS manager
FROM Employees e LEFT JOIN Employees m
ON e.manager_id = m.employee_id;
```

> The LEFT JOIN ensures that even employees without managers (e.g., CEO) are included.

### Join with Aggregation

#### Top 3 Products
**Q5:** Find the top 3 products with the highest total sales quantities in the past month.
**Tables:**
- Sales: sale_id, product_id, sale_amount, sale_date.
- Products: product_id, product_name.

```sql
SELECT p.product_id, p.product_name, SUM(s.sale_amount) AS total_sales
FROM Sales s JOIN Products p
ON s.product_id = p.product_id
WHERE s.sale_date >= DATETRUNC('month', CURRENT_DATE) - INTERVAL '1 month'
GROUP BY s.product_id, p.product_name
ORDER BY total_sales DESC
LIMIT 3;
```

```sql
WITH RecentSales AS (
    SELECT product_id, SUM(sale_amount) as total_sale_amount
    FROM Sales s
    WHERE sale_date >= DATEADD(month, -1, GETDATE())
    -- WHERE sale_date >= CURRENT_DATE - INTERVAL 1 MONTH
)
SELECT rs.product_id, p.product_name, rs.total_sale_amount
FROM RecentSales rs JOIN Products p
ON rs.product_id = p.product_id
ORDER BY rs.total_sale_amount DESC
LIMIT 3;
```

#### Orders per Customer
**Q6:** Calculate the number of orders placed by each customer per month.
**Tables:**
- `Orders`: order_id, customer_id, order_date.

```sql
SELECT c.customer_id,
    DATE_TRUNC('month', o.order_date) AS order_month,
    COUNT(o.order_id) AS total_orders
FROM Orders o INNER JOIN Customers c
ON o.customer_id = c.customer_id
GROUP BY c.customer_id, DATE_TRUNC('month', o.order_date)
ORDER BY c.customer_id, order_month;
```

#### Average Sales per Customer
**Q7:** Find the average sales amount per customer for the last 6 months.
**Tables:**
- `Sales`: sale_id, product_id, sale_amount, sale_date.
- `Customers`: customer_id, name.

```sql
SELECT c.customer_name, ROUND(AVG(s.sale_amount), 2) AS avg_sales
FROM Customers c INNER JOIN Sales s
ON c.customer_id = s.customer_id
WHERE s.sale_date >= CURRENT_DATE - INTERVAL '6 months'
GROUP BY c.customer_name
ORDER BY avg_sales DESC;
```

#### Max Daily Sales
**Q8:** Find the highest sales amount recorded for each day in the last 30 days.

```sql
SELECT sale_date, MAX(sale_amount) AS max_daily_sale
FROM Sales s
WHERE sale_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY sale_date
ORDER BY sale_date;
```

#### Percentage Contribution
**Q9:** Calculate each product’s percentage contribution to total sales.

```sql
-- Step 1: Calculate total sales and grand total
WITH ProductSales AS (
    SELECT p.product_id,
        SUM(s.sale_amount) AS total_sales,
        (SELECT SUM(sale_amount) FROM Sales) AS grand_total
    FROM Products p
    INNER JOIN Sales s
    ON p.product_id = s.product_id
    GROUP BY p.product_id
)
-- Step 2: Calculate percentage contribution
SELECT product_id, total_sales,
    ROUND((total_sales * 100.0) / grand_total, 2) AS percentage_of_total
FROM ProductSales;
```

```sql
SELECT p.product_name, SUM(s.sale_amount) AS total_sales,
    ROUND(SUM(s.sale_amount) * 100.0 / SUM(SUM(s.sale_amount)) OVER (), 2) AS percentage_of_total
FROM Products p INNER JOIN Sales s
ON p.product_id = s.product_id
GROUP BY p.product_name
ORDER BY percentage_of_total DESC;
```

## Date and Time

**Tables:**
- `Sales`: sale_id, product_id, sale_amount, sale_date.
- `Orders`: order_id, customer_id, order_date, delivery_date.
- `Holiday`: holiday_date, holiday_name.

### Last 7 Days (INTERVAL)
**Q1:** Retrieve all sales from the last 7 days

```sql
SELECT * FROM Sales s
WHERE sale_date >= CURRENT_DATE - INTERVAL '7 days';
```

### Group by Month (DATE_TRUNC, EXTRACT, DATE_PART)
**Q2:** Find total sales for each month in 2024.

```sql
SELECT 
    EXTRACT(MONTH FROM DATE_TRUNC('month', sale_date)) AS sales_month, -- Output: 1
    -- DATE_TRUNC('month', sale_date) -- Output: 2024-01-01 00:00:00
    SUM(sale_amount) AS total_sales
FROM Sales s
WHERE sale_date BETWEEN '2024-01-01' AND '2024-12-31'
GROUP BY EXTRACT(MONTH FROM DATE_TRUNC('month', sale_date))
ORDER BY sales_month;
```

**Q3:** Write a query to find the number of orders placed each month for the current year.
**Tables:**

```sql
SELECT 
    DATE_TRUNC('month', order_date) AS order_month,
    COUNT(order_id) AS total_orders
FROM Orders
WHERE DATE_PART('year', order_date) = DATE_PART('year', CURRENT_DATE)
-- WHERE EXTRACT(YEAR FROM order_date) = EXTRACT(YEAR FROM CURRENT_DATE)
GROUP BY DATE_TRUNC('month', order_date)
ORDER BY order_month;
```

### Peak Day, Hour Sales
**Q4:** Identify the day in the past month with the highest sales quantity.

```sql
SELECT sale_date, SUM(sale_amount) AS total_sales
FROM Sales
WHERE sale_date >= DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '1 month'
GROUP BY sale_date
ORDER BY total_sales DESC
LIMIT 1;

-- WHERE sale_date >= DATEADD(month, -1, GETDATE()) -- SQL Server
```

**Q5:** Find the peak hour of the day with the highest sales in the last week.

```sql
SELECT EXTRACT(HOUR FROM sale_date) AS sale_hour, 
       SUM(sale_amount) AS total_sales
FROM Sales s
WHERE sale_date >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY sale_hour
ORDER BY total_sales DESC
LIMIT 1;

-- WHERE sale_date >= DATE_TRUNC('week', CURRENT_DATE) -- INTERVAL '7 days' -- WRONG because it took the start of the week then subtract 7 days
```

### Late Deliveries
**Q6:** Find all orders where the delivery was late by more than 5 days.

```sql
SELECT *, delivery_date - order_date AS delay_days
FROM Orders
WHERE delivery_date - order_date > INTERVAL '5 days';
-- WHERE delivery_date - order_date > 5; -- if numeric value
-- WHERE DATEDIFF(delivery_date, order_date) > 5; -- MySQL
```

### Rolling Sales (OVER, ROWS, RANGE)
**Q7:** Calculate the rolling 30-day average sales amount for each day in the past 3 months.

```sql
SELECT sale_date,
    AVG(SUM(sale_amount)) OVER (
        ORDER BY sale_date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) AS rolling_avg_sales
FROM Sales s
WHERE sale_date >= CURRENT_DATE - INTERVAL '3 months'
GROUP BY sale_date
ORDER BY sale_date;
```

> First, this groups the data by sale_date and calculates the total sales for each date.
> Then, it computes the rolling average by averaging sales over a sliding 30-day window.

#### RANGE vs ROWS

```sql
SELECT sale_date,
    SUM(sale_amount) OVER (
        ORDER BY sale_date
        RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW
    ) AS rolling_sum_sales
FROM Sales
WHERE sale_date >= CURRENT_DATE - INTERVAL '3 months'
ORDER BY sale_date;
```

> The RANGE clause considers the values within the window as a range of values, not just the number of rows.

#### Other Window Frame

```sql
ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
RANGE BETWEEN CURRENT ROW AND INTERVAL '7 days' FOLLOWING
```

### Monthly Growth Rate (CTE, LAG)
**Q8:** Calculate the month-over-month sales growth for the current year.

```sql
WITH MonthlySales AS (
    SELECT DATE_TRUNC('month', sale_date) AS sale_month,
            SUM(sale_amount) AS total_sales
    FROM Sales s
    WHERE EXTRACT(YEAR FROM sale_date) = EXTRACT(YEAR FROM CURRENT_DATE)
    GROUP BY DATE_TRUNC('month', sale_date)
)
SELECT sale_month, total_sales,
        LAG(total_sales) OVER (ORDER BY month) AS previous_month_sales,
        total_sales - LAG(total_sales) OVER (ORDER BY month) AS sales_growth
FROM MonthlySales;
```

> The LAG function retrieves the total_sales from the previous month, allowing the calculation of the growth rate.

### Orders on Holidays
**Q9:** Find all orders placed on holidays or on weekends.

```sql
SELECT *
FROM Orders o LEFT JOIN Holiday h
ON o.order_date = h.holiday_date
WHERE EXTRACT(DOW FROM o.order_date) IN (0, 6) -- Sunday, Saturday
OR h.holiday_date IS NOT NULL;
```

## Subqueries
**Tables:**
- `Sales`: sale_id, customer_id, product_id, sale_date, sale_amount
- `Products`: product_id, product_name, category
- `Customers`: customer_id, customer_name
<!-- - `Orders`: order_id, customer_id, order_date -->

### Top Selling Products
**Q1:** Find the products whose total sales amount is greater than the average sales amount across all products.

```sql
SELECT p.product_name, SUM(s.sale_amount) AS total_sales
FROM Products p INNER JOIN Sales s
ON p.product_id = s.product_id
GROUP BY p.product_name
HAVING SUM(s.sale_amount) > (
    SELECT AVG(total_sales)
    FROM (SELECT SUM(sale_amount) AS total_sales
            FROM Sales
            GROUP BY product_id
         ) subquery
    );
```

### Customers with Repeat Purchases
**Q2:** Find customers who have purchased more than one product in the last 6 months.

```sql
SELECT c.customer_id, c.customer_name
FROM Customers c
WHERE c.customer_id IN (
    SELECT customer_id
    FROM Sales
    WHERE sale_date >= CURRENT_DATE - INTERVAL '6 months'
    GROUP BY customer_id
    HAVING COUNT(DISTINCT product_id) > 1
    )
ORDER BY c.customer_id, p.product_id;
```

### Products with No Sales
**Q3:** List products that have not been sold in the last 30 days.

```sql
SELECT product_id, product_name
FROM Products
WHERE product_id NOT IN (
    SELECT DISTINCT product_id
    FROM Sales
    WHERE sale_date >= CURRENT_DATE - INTERVAL '30 days'
    );
```

**Q4:** List products with zero revenue in the last 3 months.

```sql
SELECT product_id, product_name
FROM Products p INNER JOIN Sales s
ON p.product_id = s.product_id
WHERE s.sale_amount = 0;
```

### Best Selling Products per Category
**Q5:** Find the best-selling product in each category based on total sales amount.

```sql
SELECT category, product_name, total_sales
FROM (SELECT p.category, p.product_name,
        SUM(s.sale_amount) AS total_sales,
        RANK() OVER (PARTITION BY p.category ORDER BY SUM(s.sale_amount) DESC) AS rank
    FROM Products p INNER JOIN Sales s
    ON p.product_id = s.product_id
    GROUP BY p.category, p.product_name
) ranked_products
WHERE rank = 1;
```

# Python