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

### Inner Joins

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

### Left Joins

**Q2:** List all products, including those that have not been sold, with their total sales quantities.
**Tables:**
- `Products`: product_id, product_name.
- `Sales`: sale_id, product_id, quantity.

```sql
SELECT p.product_id, p.product_name,
       COALESCE(SUM(s.quantity), 0) AS total_quantity
FROM Products p LEFT JOIN Sales s
ON p.product_id = s.product_id
GROUP BY p.product_id, p.product_name;
```

> The LEFT JOIN includes all products, even those without matching sales.
> COALESCE handles NULL values by replacing them with 0 for unsold products.

### Full Outer Join
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

### Self Join
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
**Q5:** Find the top 3 products with the highest total sales quantities in the past month.
**Tables:**
- Sales: sale_id, product_id, quantity, sale_date.
- Products: product_id, product_name.

```sql
SELECT p.product_id, p.product_name, SUM(s.quantity) AS total_sales
FROM Sales s JOIN Products p
ON s.product_id = p.product_id
WHERE s.sale_date >= DATETRUNC('month', CURRENT_DATE) - INTERVAL '1 month'
GROUP BY s.product_id, p.product_name
ORDER BY total_sales DESC
LIMIT 3;
```

```sql
WITH RecentSales AS (
    SELECT product_id, SUM(quantity) as total_quantity
    FROM Sales s
    WHERE sale_date >= DATEADD(month, -1, GETDATE())
    -- WHERE sale_date >= CURRENT_DATE - INTERVAL 1 MONTH
)
SELECT rs.product_id, p.product_name, rs.total_quantity
FROM RecentSales rs JOIN Products p
ON rs.product_id = p.product_id
ORDER BY rs.total_quantity DESC
LIMIT 3;
```