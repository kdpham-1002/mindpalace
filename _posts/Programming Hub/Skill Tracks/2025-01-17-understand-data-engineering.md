---
layout: post
title: Understand Data Engineering
date: 2025-01-17 08:45 -0800
description: Data Engineering Recap
author: khoa_pham
categories: [Programming Hub, Skill Tracks]
tags: [interview preps, data roles]
pin: false
math: true
mermaid: true
toc: true
comments: true
---

## Understand Data Engineering
- **Data Workflow:** collection & storage -> preparation -> exploration & visualization -> experimentation & prediction
- Volume, Variety, Velocity, Veracity, Value (How much?, What kind?, How frequent?, How accurate?, How useful?)
- **Data Pipeline:** Ingest, Process, Store, Need pipelines, Automate flow, Provide accurate data
- Structured vs Semi-structured vs Unstructured Data
- Databases, Data Lakes > Data Warehouses; Data Catalog
- **Processing Data:** Batch (Spark, Hadoop, AWS EMR, Hive) vs Stream (Apache Nifi, Apache Storm, Kafka)
- **Scheduling:** Apache Airflow, Luigi 
- Parallel Computing; Cloud Computing
- File Storage (AWS S3, Azure Blob Storage, GC Storage)
- Computation (AWS EC2, Azure VMs, GC Engine)
- Databases (AWS RDS, Azure SQL Db, GC SQL; AWS Redshift DW, Snowflake DW, GC Datastore NoSQL)

### Learning (Postgre)SQL
- Relational Databases, Tables, Schemas, Data Types
* Queries
    * SELECT, WHERE
    * AS, DISTINCT, CREATE VIEW
* Filtering
    * AND, OR, BETWEEN, IN, (NOT) LIKE, Widcards
    * IS (NOT) NULL
* Aggregate Functions; Sorting and Grouping
    * AVG(), SUM(), MIN(), MAX(), COUNT(), ROUND()
    * ORDER BY(), GROUP BY(), HAVING
* INNER JOINs, USING
    * LEFT, RIGHT, FULL JOIN; CROSS JOIN
    * Self Joins ; Semi/ Anti Join
* UNION (ALL), INTERSECT, EXCEPT
* Subqueries inside SELECT, WHERE, FROM clauses

#### Relational Databases in SQL
- Data Types; CAST( - AS integer)
* CREATE TABLE
    * ALTER TABLE - (ADD COLUMN, RENAME COLUMN, DROP COLUMN)
    * ALTER TABLE - ALTER COLUMN - TYPE - (USING SUBSTRING firstname FROM 1 FOR 16)
    * ALTER TABLE - ALTER COLUMN - (SET NOT NULL, DROP NOT NULL)
    * INSERT INTO - SELECT DISTINCT
    * CAST(fee AS integer)
* Only contain unique values
    * ALTER TABLE - ADD CONSTRAINT uni_unq UNIQUE uni
* Primary Keys
    * ALTER TABLE - ADD COLUMN id serial
    * ALTER TABLE - ADD CONSTRAINT prof_pkey PRIMARY KEY (id)
* Surrogate Keys
    * UPDATE cars SET id = CONCAT(make, model)
    * ALTER TABLE - ADD CONSTRAINT id_pkey PRIMARY KEY (id)
* Foreign Keys
    * ALTER TABLE - RENAME COLUMN - TO uni_id
    * ALTER TABLE - ADD CONSTRAINT prof_fk FOREIGN KEY uni_id REFERENCES universities (id)
* Referential Integrity
    * ALTER TABLE - ADD CONSTRAINT - FOREIGN KEY - REFERENCES - …
    * …ON DELETE (NO ACTION, CASCADE, RESTRICT, SET NULL, SET DEFAULT)

#### Database Design
- Online Transaction Processing (OLTP - latest customer transactions) vs Online Analytical Processing (OLAP - most loyal customers)
- **Structured Data:** has schema, data types, relationships (SQL, tables in relational database)
- **Semi-Structured Data:** NoSQL, XML, JSON
- **Unstructured Data:** Photos, MP3, chat logs
- **Data Warehouses**
    - denormalized schema, dimensional modeling, MPP > Data Marts
    - Use AWS Redshift, Azure SQL DW, Gg Big Query
- **Data Lakes** 
    - cheaper; all types data
    - Use Apache Spark, Hadoop
- **ETL** (Extract, Tranform, Load)
    - (OLTP, APIs, Files, IoT Logs) -> Staging -> Data Warehouse -> ML, Data Marts, BI Tools
- **ELT** (Extract, Load, Transform)
    - (OLTP, APIs, Files, IoT Logs) -> Data Lake-> DL, Warehouse, BI Tools
- Conceptual (Entity-relational diagrams), Logical (relational model, star schema), Physical (partitions, CPUs, indexes)
- Normalization (1NF, 2NF, 3NF)
* Using views; Materialized views
    * CREATE VIEW -AS
    * GRANT UPDATE ON ratings TO PUBLIC
    * REVOKE INSERT ON films FROM db_user;
    * UPDATE - SET - WHERE
    * INSERT INTO - VALUES
    * DROP VIEW - [CASCADE \| RESTRICT]
    * CREATE OR RPLACE VIEW - AS
    * ALTER VIEW [IF EXISTS] 
    * CREATE ROLE - WITH
    * ALTER ROLE
- Vertical and Horizontal Partitioning; Sharding
- **Data Integration** - Unified Data Model
    - Sales financial data (PostgreSQL), Product behavioral data (MongoDB), Marketing contact data (CSV)
    - Transformation tools - ETL
    - Unified Data Model (RedShift)
- **DBMS types** 
    - SQL: PostgreSQL, Microsoft SQL Server, Oracle
    - NoSQL: 
        - document store (MongoDB) - content management
        - key-value store (Redis) - shopping cart online buyers
        - columnar database (Cassandra) - big data analytics for speed
        - graph database (Neo4j) - recommendations, social media data

#### Data Warehousing
- Data Lakes > Data Warehouses > Data Marts
- Business Requirements (DA, DS) -> Data Modeling (DE, DB Admin) -> ETL Design and Develop (DE, DB Admin) -> BI Application (DA/DS) -> Test & Deploy (DE)
- Architecture: Top-down (Inmon) vs Bottom-up (Kimball)
- OLAP (e.g. total amount of sales) vs OLTP (e.g. when a customer places a new order)
- Fact vs Dimension Tables
- Kimball’s 4 Step Process
- Slowly changing dimensions
- Row vs Column Store
- **ETL:** separate computer system, store transformed data, lower cost, easy for PII security because sensitive data had been excluded
- **ELT:** near real-time processes, copy of raw in DW, errors don’t require new data pulls
- On-premise vs Cloud (AWS Redshift, Azure Synapse Analytics, Snowflake, GBQ)

#### Learning Snowflake(SQL)
- Architecture and Competitors (Databricks, AWS Redshift, Google Big Query)
* Advanced SQL Concepts   
    * NATURAL, LATERAL JOINs
    * Subqueries and CTEs
    * VARIANT data type; Handling JSONs
        * PARSE_JSON , OBJECT_CONSTRUCT

### Learning Python
- Variables, Data Types
* Lists, Dictionaries, Sets and Tuples
* IF - ELSE, and, or, (not) in; FOR, range(); WHILE, break
- Functions, Modules, Packages
- Docstrings
- Default, Keyword, Arbitrary Arguments (*args, **kwargs)
* Lambda Functions - `(lambda x, y: x**y)(2, 3)`
* Errors
    * TypeError, ValueError, Tracebacks
    * try-except; raise

#### Importing Data & Cleaning Data
- Reading, writing to a text file
* Importing using Numpy, Pandas
    * np. loadtxt(filename, delimiter= ‘ , ’ ,  skiprows=1, usecols=[0, 2])
    * pd.read_csv(filename)
    * pd.ExcelFile(filename)
* SQL querying in Python; with pandas
```python
from sqlalchemy import create_engine 
    engine = create_engine('sqlite:///db.sqlite')
```
* HTTP requests
```python
from urllib.request import urlretrieve, urlopen, Request
# --- or ---
import requests
```
* Scraping the Web
```python
from bs4 import BeautifulSoup
import requests
```
* JSONs, APIs -> practice Twitter API

#### Data Ingestion with Pandas

#### ETL & ELT with Python

### Learning Git

### Learning Apache Airflow

### Cloud Computing
- On-premise vs Cloud (IaaS vs PaaS vs SaaS)
- AWS (Cloud - S3, EC2, RDS; Data - Redshift, Kinesis, SageMaker)
- Azure (Cloud - Blob Storage, VMs, SQL Db; Data - Data Lake Storage, Stream Analytics, Machine Learning)
- Google (Cloud - GC Storage, GC Compute Engine, GC SQL; Data - Big Query, Dataflow, AutoML) 


### Containerization, Virtualization and Orchestration
- Docker, Kubernetes