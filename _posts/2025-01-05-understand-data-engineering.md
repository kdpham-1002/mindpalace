---
layout: post
title: Understand Data Engineering
date: 2025-01-05 08:45 -0800
description: What to learn in Data Engineering
author: khoa_pham
categories: [Study Workspace, Coding]
tags: [study notes, data engineering]
pin: true
math: true
mermaid: true
toc: true
comments: true
---

## Understand Data Engineering

### Data Engineering Basic Concepts
- **Data Workflow:** collection & storage -> preparation -> exploration & visualization -> experimentation & prediction
- Volume, Variety, Velocity, Veracity, Value (How much?, What kind?, How frequent?, How accurate?, How useful?)
- **Data Pipeline:** Ingest, Process, Store, Need pipelines, Automate flow, Provide accurate data
- Structured vs Semi-structured vs Unstructured Data
- Databases, Data Lakes > Data Warehouses; Data Catalog
- **Processing Data:** Batch (Spark, Hadoop, AWS EMR, Hive) vs Stream (Apache Nifi, Apache Storm, Kafka)
- **Scheduling:** Apache Airflow
- Parallel Computing; Cloud Computing
- File Storage (AWS S3, Azure Blob Storage, GC Storage)
- Computation (AWS EC2, Azure VMs, GC Engine)
- Databases (AWS RDS, Azure SQL Db, GC SQL; AWS Redshift DW, Snowflake DW, GC Datastore NoSQL)

### Intro to SQL
- Relational Databases, Tables, Data Types
* AS, DISTINCT, CREATE VIEW
    - SELECT, WHERE
    - AND, OR, BETWEEN, IN, (NOT) LIKE
    - AVG(), SUM(), MIN(), MAX(), COUNT(), ROUND()
    - ORDER BY(), GROUP BY(), HAVING
* INNER JOIN, USING, AND
    - LEFT, RIGHT, FULL JOIN; CROSS JOIN
    - Self Join; Semi/ Anti Join
    - UNION (ALL), INTERSECT, EXCEPT
* Subqueries inside SELECT, WHERE, FROM clauses

### Project: Analyzing Students’ Mental Health

### Intro to Relational Databases
- Query information_schema
```SQL
SELECT constraint_name, table_name, constraint_type
FROM information_schema.table_constraints
WHERE constraint_type = 'FOREIGN KEY';
```
* CREATE TABLE
    - ALTER TABLE - (ADD COLUMN, RENAME COLUMN, DROP COLUMN)
    - ALTER TABLE - ALTER COLUMN - TYPE - (USING SUBSTRING firstname FROM 1 FOR 16)
    - ALTER TABLE - ALTER COLUMN - (SET NOT NULL, DROP NOT NULL)
    - INSERT INTO - SELECT DISTINCT
    - CAST(fee AS integer)
* Only contain unique values
    - ALTER TABLE - ADD CONSTRAINT uni_unq UNIQUE uni
* Primary Keys
    - ALTER TABLE - ADD COLUMN id serial
    - ALTER TABLE - ADD CONSTRAINT prof_pkey PRIMARY KEY (id)
* Surrogate Keys
    - UPDATE cars SET id = CONCAT(make, model)
    - ALTER TABLE - ADD CONSTRAINT id_pkey PRIMARY KEY (id)
* Foreign Keys
    - ALTER TABLE - RENAME COLUMN - TO uni_id
    - ALTER TABLE - ADD CONSTRAINT prof_fk FOREIGN KEY uni_id REFERENCES universities (id)
* Referential Integrity
    - ALTER TABLE - ADD CONSTRAINT - FOREIGN KEY - REFERENCES - …
    - …ON DELETE (NO ACTION, CASCADE, RESTRICT, SET NULL, SET DEFAULT)

### Database Design
- Online Transaction Processing (OLTP - latest customer transactions) vs Online Analytical Processing (OLAP - most loyal customers)
- **Structured Data:** has schema, data types, relationships (SQL, tables in relational database)
- **Semi-Structured Data:** NoSQL, XML, JSON
- **Unstructured Data:** Photos, MP3, chat logs
* **Data Warehouses**
    - denormalized schema, dimensional modeling, MPP > Data Marts
    - Use AWS Redshift, Azure SQL DW, Gg Big Query
* **Data Lakes** 
    - cheaper; all types data
    - Use Apache Spark, Hadoop
* **ETL** (Extract, Tranform, Load)
    - (OLTP, APIs, Files, IoT Logs) -> Staging -> Data Warehouse -> ML, Data Marts, BI Tools
* **ELT** (Extract, Load, Transform)
    - (OLTP, APIs, Files, IoT Logs) -> Data Lake-> DL, Warehouse, BI Tools