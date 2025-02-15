---
layout: post
title: MySQL Technical Practice Questions
date: 2025-01-06 21:13 -0800
description: Interview questions for MySQL
author: khoa_pham
categories: [Programming Hub, Career Preps]
tags: [SQL, interview preps, coding]
pin: false
math: true
mermaid: true
toc: true
comments: true
---

### Q1: UNIQUE records

> Q1: How do you select **unique** records from a table?

* Using the DISTINCT clause, we can select unique records from a table.
* With multiple columns, it considers the combination of values in those columns to determine uniqueness.

**For example:**  
101 - John  
101 - Joe  
102 - Jay  

### Q2: TABLE vs. VIEW

> Q2: What is the difference between a **TABLE** and a **VIEW**?

* A table stores data (in memory) and can be modified, whereas a view does not store data and cannot be modified directly.
* The data in a view is derived from one or more tables and is only accessible through the SELECT statement that defines the view.
* A table has its own data, which is stored on disk, while a view is simply a SELECT statement that is run every time the view is queried.

**Bonus:** Views can provide a performance benefit by allowing you to create a simplified version of a table or by allowing you to join multiple tables together into a single virtual table. Views can be very helpful when looking at more complex queries that you are going to reuse over and over. You won’t have to rewrite the query or join.

### Q3: INNER JOIN vs OUTER JOIN

> Q3: What is the difference between **INNER JOIN** and **OUTER JOIN**?

* An **Inner Join** returns only the rows that have matching values in both tables. It only returns the rows where the join condition is true.

* A **Left Join** (Left Outer Join) returns all rows from the left table and the matched rows from the right table. If there is no match, NULL values are
returned for the right table's columns.

* A **Right Join** (Right Outer Join) returns all rows from the right table and the matched rows from the left table. If there is no match, NULL values are
returned for the left table's columns.

* A **Full Join** (Full Outer Join) returns all rows from both tables and the matched rows from both tables. If there is no match, NULL values are returned
for the non-matching side's columns.

Now of course we don't have Full Outer Joins in MySQL, but it is present in other types of SQL so it's worth noting in an interview. This will show your
understanding of different types of SQL.

### Q4: GROUP BY and Aggregate Functions

> Q4: How does **GROUP BY** work? What are **aggregate functions**?

* Group By is used to group rows in a result set based on one or more columns
* Aggregate functions are functions that allow you to perform a calculation on a set of values and return a single value
* When these two are combined you can run aggregate functions on grouped rows and perform calculations on all the rows in each group

**Bonus:** If you then want to filter on those aggregations, you can use the Having Clause.

You can mention the reason we need to use the Having Clause is because of the execution order on the back end of a query. The where clause is run before the Group By so we cannot filter grouped rows until after the Group By. This is why we use Having which comes after Group By.

### Q5: WHERE vs HAVING

> Q5: What is the difference between **WHERE** and **HAVING** in a query?

* The WHERE clause is used to filter rows before the group by operation, it filters rows individual records, which results in a smaller set of data.
WHERE is used to filter data based on specific conditions, such as column values, data ranges, and null values.
* The HAVING clause is used to filter rows after the group by operation. It filters groups of data based on aggregate function values.

### Q6: CASE Statement

> Q6: What is a **CASE** statement?

* The CASE Statement allows you to create a logic in the Select statement that performs different actions based on specified conditions.
* It is like an if/else statement in most programming languages.
* We can use it by saying CASE WHEN. This is where you specify your conditions.
* If the conditions are met, we can perform an action by saying THEN and specifying what should happen.
* We also have an optional ELSE Statement which executes if none of the WHEN/THEN statements evaluate to True.
* END will close out the CASE Statement.

**Bonus:** Case Statements are great to use to categorize use or to perform calculations based on specific conditions.

### Q7: PRIMARY KEY vs FOREIGN KEY

> Q7: What is the difference between **PRIMARY KEY** and **FOREIGN KEY**?

* Both are types of constraints that are used to define the relationships between tables in a relational database.
* A Primary Key is a unique identifier for each row in a table.
* A Foreign Key is a column in a table that is used to establish a link between data in two tables.

**Bonus:** Foreign keys ensure referential integrity between tables. This means that you cannot insert a row with a foreign key value that does not exist in the primary key table. If a row in the primary key table is deleted or updated, corresponding rows in other tables that have a foreign key relationship with the primary key table are also updated or deleted.
You can also mention Data Modeling. Primary and Foreign Keys are used when creating the data schemas and also when modeling data in a database.

### Q8: JOIN vs UNION 

> Q8: What is the difference between **JOIN** and **UNION**?

* A **JOIN** combines rows from two or more tables based on a related column between them. The related column is usually the primary key of one table, and a foreign key in the other table.
* A **UNION** combines a result set of two or more queries (Select Statements).
* Joining appends horizontally while using a Union appends vertically. 
 
**Bonus:** A UNION by default removes duplicates, but you can use UNION ALL to keep duplicates in the output

### Q9: Standardizing Data

> Q9: How do you **standardize** data in SQL?

* Standardizing data in SQL involves ensuring that data is consistent and conforms to a set of rules or standards.
* This is usually done as part of the data cleaning process by removing inaccuracies, inconsistencies, or duplicate data from the data
* An example would be if we had several different date formats in a single column. Standardizing the data would be changing them to all be the same format.

**Bonus:** Standardization of the data is not a one-time process – it happens over time. It’s also something that’s highly unique to each dataset and how it is standardized depends on the data.

### Q10: Data Collection Process

> Q10: Describe your **Data Collection** process

1. Meet with Stakeholders and Data Owners
2. Connect to Source data and Extract the data
3. Transform the data based on business rules (data cleaning, standardization, etc)
4. Load data into a Staging database
5. Transfer to Production database

**Bonus:** Understanding the data and how it's collected at the source is important to understanding the business rules you'll create to transform the data.
Business Rules are the rules that you create in order to transform the data during the ETL Process. Usually Business rules are applied to individual
columns, so there are a lot of business rules that are created for each dataset.

### Q11: Data Cleaning Process

> Q11: Describe your **Data Cleaning** process

1. Removing duplicate records
2. Correcting inaccuracies or inconsistencies in the data
3. Removing outliers or incorrect values
4. Fixing or removing invalid or missing data
5. Standardizing data (such as date or currency format)
6. Identifying and removing any errors or inconsistencies in the data
7. Merge similar data (merging employee names that are misspelled or inconsistent)

**Bonus:** Data Cleaning is a cycle. This means you don't just clean the data once. As new quality issues arise in the data, new business rules need to be
created to clean the data further.
Knowing how the data is collected at the source is extremely important. This makes all the difference in fully understanding what business rules you may
need to create. This is part of having domain knowledge and talking with clients/stakeholders to understand their entire processes.

### Q12: Query Syntax vs Execution

> Q12: Describe how a query is written (Syntax) vs how it is executed?

The typical Query is written in this order:

`SELECT, FROM, WHERE, GROUP BY, HAVING, ORDER BY, and LIMIT`

How it executes on the backend is different, it will usually run in this order:

`FROM, WHERE, GROUP BY, HAVING, SELECT, ORDER BY, and LIMIT`

The biggest difference is the SELECT Statement in the execution order is much lower. This is why we need a HAVING clause. If we use GROUP BY, we can’t filter on it yet because it hasn’t been grouped in the execution.

### Q13: Handling NULL values

> Q13: How do you handle **NULL** values?

How you handle NULL values depends greatly on what you want to do with them and their intended purpose. Here are a few ways to handle NULL Values:
1. You can simply filter out NULL values using the IS NULL or IS NOT NULL operators in the WHERE Statement
2. Replace NULL values. Depending on the data it may make sense to replace data with “0”s or (Mean or Median) averages of the field
3. Use functions to utilize NULL Values like COALESCE, NULLIF, ISNULL, and IFNULL

### Q14: Difficult Datasets

> Q14: What makes a dataset 'easy' or 'difficult' to work with?

* A dataset may be considered easy when the hard work is done upfront. It's cleaned properly, data is standardized, columns are named appropriately,
and others factors. These things make it easy to manipulate and drive insights.
* A difficult dataset is the opposite. It's not straightforward, relationships between tables have not been defined properly, it's not fully cleaned, and it
doesn't have intuitive column names.

**Bonus:** Sometimes this is something that needs to be revised in the Data Collection process, but sometimes it's in how the data is created on the client
side. This is something a data analyst would need to determine.

### Q15: Query Optimization

> Q15: What do you know about **Query Optimization**?

Optimizing a query involves making changes to the query and the database to make it run faster and more efficiently.
Ways to optimize queries:
1. Use Indexes to provide faster access to the data
2. Limit the number of subqueries and joins in a single query and use them wisely.
3. Optimize datatypes to only store what you need to store in the most efficient way.
4. Use LIMIT and don't use SELECT * to reduce the volume in the output