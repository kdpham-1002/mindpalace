---
layout: post
title: Amazon BIE/DE Practice Questions
date: 2025-01-19 14:30 -0800
description: Practice Amazon technical interview questions
author: khoa_pham
categories: [Programming Hub, Career Preps]
tags: [interview preps, coding, SQL, python]
pin: false
math: true
mermaid: true
toc: true
comments: true
---

> Link to [Amazon BIE/DE Description Requirements](https://khoapham1002.github.io/mindpalace/posts/amazon-bie-de-requirements/)

## SQL  
### Joins  
**Tables:**
- `Orders`: order_id, customer_id, order_date.
- `Customers`: customer_id, name, city.
- `Sales`: sale_id, product_id, sale_amount.
- `Suppliers`: supplier_id, supplier_name.
- `Products`: product_id, product_name, supplier_id.
- `Employees`: employee_id, name, manager_id.

#### All Orders (Inner Joins)
**Q1:** Write a query to find all orders placed by customers who live in New York.

```sql
SELECT o.order_id, o.customer_id, o.order_date, c.name
FROM Orders o INNER JOIN Customers c
ON o.customer_id = c.customer_id
WHERE c.city = 'New York';
```

> The INNER JOIN ensures only rows with matching customer_id in both tables are included.

#### All Products (Left Joins)
**Q2:** List all products, including those that have not been sold, with their total sales quantities.

```sql
SELECT p.product_id, p.product_name,
       COALESCE(SUM(s.sale_amount), 0) AS total_sale_amount
FROM Products p LEFT JOIN Sales s
ON p.product_id = s.product_id
GROUP BY p.product_id, p.product_name;
```

> The LEFT JOIN includes all products, even those without matching sales.
> COALESCE handles NULL values by replacing them with 0 for unsold products.

#### All Suppliers & Products (Full Outer Join)
**Q3:** Combine data from Suppliers and Products, showing all products and suppliers, even if there are no matches.

```sql
SELECT 
FROM Suppliers s FULL OUTER JOIN Products p
ON s.supplier_id = p.supplier_id;
```

> The FULL OUTER JOIN ensures inclusion of all rows from both tables, with NULL where no matches exist.

#### All Employees (Self Join)
**Q4:** List all employees and their managers.

```sql
SELECT e.name AS employee, m.name AS manager
FROM Employees e LEFT JOIN Employees m
ON e.manager_id = m.employee_id;
```

> The LEFT JOIN ensures that even employees without managers (e.g., CEO) are included.

#### Join with Aggregation

**Tables:**
- `Sales`: sale_id, product_id, sale_amount, sale_date.
- `Customers`: customer_id, name.
- `Orders`: order_id, customer_id, order_date.
- `Products`: product_id, product_name.

##### Top 3 Products
**Q5:** Find the top 3 products with the highest total sales quantities in the past month.

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

##### Orders per Customer
**Q6:** Calculate the number of orders placed by each customer per month.

```sql
SELECT c.customer_id,
    DATE_TRUNC('month', o.order_date) AS order_month,
    COUNT(o.order_id) AS total_orders
FROM Orders o INNER JOIN Customers c
ON o.customer_id = c.customer_id
GROUP BY c.customer_id, DATE_TRUNC('month', o.order_date)
ORDER BY c.customer_id, order_month;
```

##### Average Sales per Customer
**Q7:** Find the average sales amount per customer for the last 6 months.

```sql
SELECT c.customer_name, ROUND(AVG(s.sale_amount), 2) AS avg_sales
FROM Customers c INNER JOIN Sales s
ON c.customer_id = s.customer_id
WHERE s.sale_date >= CURRENT_DATE - INTERVAL '6 months'
GROUP BY c.customer_name
ORDER BY avg_sales DESC;
```

##### Max Daily Sales
**Q8:** Find the highest sales amount recorded for each day in the last 30 days.

```sql
SELECT sale_date, MAX(sale_amount) AS max_daily_sale
FROM Sales s
WHERE sale_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY sale_date
ORDER BY sale_date;
```

**Output:**
product_name | total_sales | percentage_of_total
:--:|:---:|:--:
Product A | 300 | 60
Product B | 200 | 40

### Date and Time

**Tables:**
- `Sales`: sale_id, product_id, sale_amount, sale_date.
- `Orders`: order_id, customer_id, order_date, delivery_date.
- `Holiday`: holiday_date, holiday_name.

#### Last 7 Days (INTERVAL)
**Q1:** Retrieve all sales from the last 7 days

```sql
SELECT * FROM Sales s
WHERE sale_date >= CURRENT_DATE - INTERVAL '7 days';
```

#### Group by Month (DATE_TRUNC, EXTRACT, DATE_PART)
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

#### Peak Day, Hour Sales
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

#### Late Deliveries
**Q6:** Find all orders where the delivery was late by more than 5 days.

```sql
SELECT *, delivery_date - order_date AS delay_days
FROM Orders
WHERE delivery_date - order_date > INTERVAL '5 days';
-- WHERE delivery_date - order_date > 5; -- if numeric value
-- WHERE DATEDIFF(delivery_date, order_date) > 5; -- MySQL
```

#### Rolling Sales (OVER, ROWS, RANGE)
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

######## RANGE vs ROWS

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

#### Monthly Growth Rate (CTE, LAG)
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

#### Orders on Holidays
**Q9:** Find all orders placed on holidays or on weekends.

```sql
SELECT *
FROM Orders o LEFT JOIN Holiday h
ON o.order_date = h.holiday_date
WHERE EXTRACT(DOW FROM o.order_date) IN (0, 6) -- Sunday, Saturday
OR h.holiday_date IS NOT NULL;
```

### Subqueries
**Tables:**
- `Sales`: sale_id, customer_id, product_id, sale_date, sale_amount
- `Products`: product_id, product_name, category
- `Customers`: customer_id, customer_name
<!-- - `Orders`: order_id, customer_id, order_date -->

#### Top Selling Products
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

#### Customers with Repeat Purchases
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

#### Products with No Sales
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

#### Best Selling Products per Category
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

```sql
SELECT p.category, p.product_name,
    SUM(s.sale_amount) AS total_sales
FROM Products p INNER JOIN Sales s
ON p.product_id = s.product_id
GROUP BY p.category, p.product_name
HAVING SUM(s.sale_amount) = (
    SELECT MAX(total_sales)
    FROM (SELECT p.category, SUM(s.sale_amount) AS total_sales
            FROM Products p INNER JOIN Sales s
            ON p.product_id = s.product_id
            GROUP BY p.category) AS subquery
    WHERE subquery.category = p.category
);
```

#### Customers with High Spending
**Q6:** Find the top 5 customers by their total purchase amount.

```sql
SELECT customer_id, customer_name, total_purchases
FROM (SELECT c.customer_id, c.customer_name,
        SUM(s.sale_amount) as total_purchases,
        RANK() OVER (ORDER BY SUM(s.sale_amount) DESC) AS rank
        FROM Customers c INNER JOIN Sales s
        ON c.customer_id = s.customer_id
        GROUP BY c.customer_id, c.customer_name) AS ranked_customers
WHERE rank <= 5;
```

```sql
SELECT c.customer_id, c.customer_name,
    SUM(s.sale_amount) AS total_purchases
FROM Customers c INNER JOIN Sales s
ON c.customer_id = s.customer_id
GROUP BY c.customer_id, c.customer_name
ORDER BY SUM(s.sale_amount) DESC
LIMIT 5;
```

#### Sales Across Categories
**Q7:** Find categories where sales exceeded the average sales.

```sql
SELECT p.category, SUM(s.sale_amount) AS total_sales
FROM Sales s INNER JOIN Products p
ON s.product_id = p.product_id
GROUP BY p.category
HAVING SUM(s.sale_amount) > (
    SELECT AVG(total_sales)
    FROM (SELECT SUM(sale_amount) AS total_sales
            FROM Sales
            GROUP BY product_id) AS subquery
);
```

```sql
WITH CategorySales AS (
    SELECT p.category, SUM(s.sale_amount) AS total_sales
    FROM Sales s INNER JOIN Products p
    ON s.product_id = p.product_id
    GROUP BY p.category
),
AverageSales AS (
    SELECT AVG(total_sales) AS avg_sales
    FROM CategorySales
)
SELECT category, total_sales
FROM CategorySales
WHERE total_sales > (SELECT avg_sales FROM AverageSales);
```

### Window Functions

**Tables:**
- `Employees`: employee_id, emp_name, department_id, salary
- `Sales`: sale_id, product_id, sale_amount, sale_date
- `Orders`: order_id, customer_id, order_date

#### Rank Employees by Salary
**Q1:** Find the rank of each employee by their salary within their department.

```sql
SELECT department_id, emp_name, salary,
    RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) AS salary_rank
FROM Employees;
```

#### Running Total Sales by Product
**Q2:** For each product, calculate the cumulative sales over time.

```sql
SELECT product_id, sale_date, sale_amount,
    SUM(sale_amount) OVER (PARTITION BY product_id ORDER BY sale_date) AS cumulative_sales
FROM Sales;
```

#### First and Last Orders
**Q3:** Find the first and last order date for each customer.

```sql
SELECT customer_id, order_date,
    FIRST_VALUE(order_date) OVER (PARTITION BY customer_id ORDER BY order_date) AS first_order,
    LAST_VALUE(order_date) OVER (PARTITION BY customer_id ORDER BY order_date) AS last_order
FROM Orders;
```

#### Flag Highest Sales
**Q4:** Add a column to flag the highest sale for each product.

```sql
SELECT product_id, sale_date, sale_amount,
    CASE WHEN sale_amount = MAX(sale_amount) 
        OVER (PARTITION BY product_id)
        THEN 'Highest Sale' ELSE 'Regular Sale' 
    END AS sale_flag
FROM Sales;
```

```sql
SELECT product_id, sale_date, sale_amount,
    CASE WHEN sale_amount = MAX(sale_amount) 
        OVER (PARTITION BY product_id) THEN 'Highest Sale' 
        ELSE 'Regular Sale' 
    END AS sale_flag
FROM Sales;
```

#### Median Sales Amount
**Q5:** Find the median sales amount for each product.

```sql
SELECT product_id, sale_date, sale_amount,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY sale_amount) 
    OVER (PARTITION BY product_id) AS median_sales
FROM Sales;
```

#### Percentage Contribution
**Q6a:** Calculate each product’s sales as a percentage of total sales.

```sql
SELECT product_name, sale_amount,
    -- ROUND(SUM(sale_amount) OVER (PARTITION BY product_name) * 100.0 / SUM(sale_amount) OVER (), 2) AS percent_of_total
    ROUND(sale_amount * 100.0 / SUM(sale_amount) OVER (), 2) AS percent_of_total
FROM Sales;
```

**Output:**
product_name | sale_amount | percent_of_total
:--:|:---:|:--:
A | 100 | 20
A | 200 | 40
B | 150 | 30
B | 50  | 10

**Q6b:** Calculate each product’s percentage contribution to total sales.

```sql
SELECT p.product_name, SUM(s.sale_amount) AS total_sales,
    ROUND(SUM(s.sale_amount) * 100.0 / SUM(SUM(s.sale_amount)) OVER (), 2) AS percentage_of_total
FROM Products p INNER JOIN Sales s
ON p.product_id = s.product_id
GROUP BY p.product_name
ORDER BY percentage_of_total DESC;
```

**Output:**
product_name | total_sales | percentage_of_total
:--:|:---:|:--:
Product A | 300 | 60
Product B | 200 | 40

-- Q6b Method 2 --
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

## Data Modeling

### Star Schema Design
You are tasked with designing a data warehouse for an online retail store. The business needs to track:
1. sales_amount, units_sold, and discount.
2. Products with attributes such as product_name and category.
3. Customers with attributes such as customer_name and region.
4. Dates with attributes such as day, month, and year.

***Solution:*** 
1. Fact Table:  

`Sales`     
| sales_id | product_id | customer_id | date_id | sales_amount | units_sold | discount |

2. Dimension Tables:

`Products`   
| product_id | product_name | category |

`Customers`   
| customer_id | customer_name | region |

`Date`  
| date_id | day | month | year |

***Benefits of Star Schema:***
•	Simplifies querying with fewer joins.
•	Optimized for OLAP systems.

### Snowflake Schema Design
Normalize the Products dimension from the previous exercise into a snowflake schema.

***Solution:*** 
1. Updated `Products` Dimension:

`Products`   
| product_id | product_name | category_id |

2. New Dimension Table:  

`Categories`  
| category_id | category_name |

### Fact Table Design
Design a fact table to track daily website traffic metrics, including:
1. page_views, unique_visitors, and average_session_duration.  
2. Metrics should be tracked by date, page, and region.  

***Solution:***  
1. Fact Table:

`Website_Traffic`  
| traffic_id | date_id | page_id | region_id | page_views | unique_visitors | avg_session_duration |

2. Dimension Tables:  
`Date`: date_id, day, month, year.  
`Pages`: page_id, page_name, category.  
`Region`: region_id, region_name, country.


## ETL Workflow

### Data Cleaning
- Missing Values
- Duplicate Records

### Data Loading
### Data Transformation
### Error Handling


## AWS Tools

### Amazon Redshift (Data Warehousing)
- How to load data into Redshift using COPY commands from S3.
- Writing queries in Redshift (optimized SQL for large datasets).
- Distribution styles (key, all, even) and sort keys for performance.
- Redshift Spectrum (querying S3 data without importing it into Redshift).

    - How would you optimize a Redshift table for query performance?
    - What’s the difference between distribution styles in Redshift?

### Amazon S3 (Data Storage)
- Uploading and managing data in S3 buckets.
- Data formats: CSV, JSON, Parquet (important for analytics).
- Integrating S3 with other tools like Redshift and Athena.

    - How would you structure data in an S3 bucket for an ETL pipeline?
    - Why might you choose Parquet over CSV for storing data in S3?

### AWS Glue (ETL Service)
- Creating Glue jobs to transform data (Python/Scala scripts).
- Using Glue Data Catalog to define schemas for your datasets.
- Crawlers to automatically detect data structure.
- Connections to S3, Redshift, and other data sources.

    - How would you use Glue to transform raw JSON data into a structured format?
    - Explain the role of Glue Data Catalog in an ETL pipeline.

### Amazon Athena (Querying S3 Data)
- Writing queries to analyze data stored in S3.
- Supported file formats (e.g., Parquet, ORC) for efficient queries.
- Integration with Glue Data Catalog for metadata management.

    - How would you query large CSV files in S3 without loading them into a database?
    - What are the advantages of using Parquet with Athena?

### Amazon Quicksight (BI Tool)
- Creating data sources and datasets in QuickSight.
- Building interactive dashboards and defining KPIs.
- Embedding QuickSight dashboards in applications.

    - How would you create a dashboard to show sales trends over time?
    - What steps are involved in connecting QuickSight to Redshift?

### AWS Lambda (Serverless Compute)
- Writing Lambda functions for lightweight ETL tasks.
- Triggering Lambda from S3 events or scheduled CloudWatch events.

    - How would you use Lambda to process and clean data uploaded to an S3 bucket?
    - Explain the benefits of using Lambda over a traditional ETL tool.