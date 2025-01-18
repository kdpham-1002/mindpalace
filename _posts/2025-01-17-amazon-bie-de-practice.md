---
layout: post
title: Amazon BIE/DE Practice
date: 2025-01-17 14:31 -0800
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
