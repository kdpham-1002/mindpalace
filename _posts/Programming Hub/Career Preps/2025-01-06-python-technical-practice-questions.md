---
layout: post
title: Python Technical Practice Questions
date: 2025-01-06 21:13 -0800
description: Interview questions for Python
author: khoa_pham
categories: [Programming Hub, Career Preps]
tags: [python, interview preps, coding]
pin: false
math: true
mermaid: true
toc: true
comments: true
---

### Q1: Series vs Dataframe

> Q1: What is the difference between a **series** and a **dataframe** in Pandas?

* Series is one-dimensional, while DataFrame is two-dimensional
* Series can be seen as a single column of data, while DataFrame is a combination of multiple Series.

**Bonus:** When working with DataFrames you'll often select a single Series to manipulate and join back to the original DataFrame.

### Q2: loc vs iloc

> Q2: What is the difference between the **loc** and **iloc** methods in Pandas?

* "iloc" is used for ***integer-location*** based indexing/selection by position. This means that you use the integer
index of the row or column you want to access (starting from 0).
* "loc" is used for ***label-location*** based indexer for selection by
label. This means that you use the named index of the row
or column you want to access.

### Q3: axis parameter

> Q3: Explain the use of **'axis ='** parameter in Pandas?

* axis=0 will be applied along the "row" direction
* axis=1 will be applied along the "column" direction

### Q4: inplace parameter

> Q4: Explain the use of **'inplace ='** parameter in Pandas?

* If you perform an operation on a DataFrame and don't use
inplace, a modified copy of that DataFrame is created.
* If you set ***'inplace = True'***, the operation will modify the
DataFrame itself

### Q5: Missing data

> Q5: How can you handle missing data?

1. Drop Missing Values using "dropna()"
2. Fill missing Values with specific values using "fillna()"
3. Fill missing Values with computer values (mean, median, mode) using "fillna().mean()"
4. Using the Interopolation with interopolatel() method

**Bonus:** Interopolation is especially useful with time series data and estimates values between two known values

### Q6: Large data sets

> Q6: How can you handle very large data sets that don't fit into memory?

You can handle very large data sets that don't fit into memory using several techniques:
1.	Process Data in Chunks  
•	Why: Instead of loading a huge file all at once, you process smaller parts (chunks) of it so you don’t overload your memory.  
•	Pandas (use read_csv(..., chunksize=...))  

2.	Sample for Exploration  
•	Why: You can examine a smaller, representative subset of data for quick analysis before working with the full dataset.  
•	Pandas (e.g., df.sample(n=1000))  
•	SQL (TABLESAMPLE clause in some databases)  

3.	Store Data in a Database  
•	Why: Databases efficiently handle large datasets and let you query only the parts you need.  
•	MySQL, PostgreSQL, SQLite (Databases)  
•	SQLAlchemy or pandas read_sql_query (to connect Python/Pandas with databases)  

4.	Use Distributed Computing  
•	Why: If your dataset is extremely large, chunking might still be too slow. A cluster of machines can split up the work.  
•	Apache Spark (via PySpark) automatically distributes your data across a cluster and processes it in parallel.  
•	Dask offers a more “Pandas-like” interface but still runs across multiple machines or cores.  
•   Databricks: A managed platform that makes it easier to run Spark in the cloud without handling cluster setup yourself.

5.	Use Efficient File Formats  
•	Why: Formats like Parquet or ORC store data in a compressed, columnar layout, so you read less from disk and load only what you need.  
•	Parquet, ORC  
•	pyarrow or fastparquet (for reading/writing parquet in Python)  

6.	Stream Data in Real Time  
•	Why: If data arrives continuously (like from sensors or a website), you process records on the fly instead of storing it all first.  
•	Kafka, RabbitMQ (for real-time data pipelines)  
•	Spark Streaming, Apache Flink, Dask (for streaming computation)  

### Q7: Handling New Data

> Q7: What steps would you take to analyze a new, unfamiliar dataset?

Here are the steps I usually take:
1. Importing Data into a Dataframe
2. Preview data using head(), tail(), info(), describe()
3. Check for missing values
4. Data Cleaning & Feature Engineering
5. Exploratory Data Analysis

### Q8: Favorite Pandas Trick

> Q8: What is your favorite "trick" function in Pandas that you find particularly useful?

#### .query() method

The query() method allows you to filter a DataFrame using a string expression. It’s often more readable than using standard boolean indexing.

Pro tip: You can reference columns without quoting them as df['column']; just type the column name in your query string.

```python
import pandas as pd

df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [24, 42, 18, 35],
    'City': ['NY', 'LA', 'NY', 'Chicago']
})

## Using query() to filter rows where Age > 20 and City == 'NY'
filtered_df = df.query("Age > 20 and City == 'NY'")
# filtered_df = df[(df['Age'] > 20) & (df['City'] == 'NY')]
print(filtered_df)
```

#### .eval() method

The eval() lets you evaluate expressions on columns in a DataFrame. This can be particularly helpful for performing multiple operations at once or when building dynamic expressions.

Pro tip: You can also create multiple columns in one go using a single eval statement.

```python
import pandas as pd

df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

# Evaluate an expression and create a new column
# df.eval("C = A + B", inplace=True)
# df.eval("D = A * B", inplace=True)
df.eval("""
    C = A + B
    D = A * B
""", inplace=True)
print(df, "\n")

# Evaluate a complex expression
result = df.eval("(A + B) / C")
print(result)
```

#### .assign() method

The assign() makes it convenient to add new columns (or transform existing ones) without breaking up your method chain. You can keep your transformations “flowing” in a single pipeline.

```python
import pandas as pd

df = pd.DataFrame({
    'X': [10, 20, 30],
    'Y': [2, 4, 8]
})

# def calculate_z(df):
#     return df['X'] * 2
# def calculate_xy_sum(df):
#     return df['X'] + df['Y']

# df_assigned = df.assign(
#     Z=calculate_z,
#     XY_sum=calculate_xy_sum
# )


## Create new columns directly
df_assigned = df.assign(
    Z=lambda df: df['X'] * 2,   # Using a lambda
    XY_sum=lambda df: df['X'] + df['Y']
)
print(df_assigned)
```

#### .explode() method

When you have a column of lists (or arrays), explode() will unnest those lists into multiple rows, duplicating the other row values as needed.

```python
import pandas as pd

df = pd.DataFrame({
    'Id': [1, 2, 3],
    'Colors': [
        ['Red', 'Blue'],
        ['Green', 'Yellow'],
        ['Pink']
    ]
})

# Explode the list in "Colors" into separate rows
exploded_df = df.explode('Colors')
print(exploded_df)
```

#### .cut() method

The cut() is typically used when you have numeric data and you want to split (or “bin”) it into intervals that you define. Think of it as creating buckets for your data values.

Pro tip: If you don’t specify labels, cut() will create them automatically (like (0, 3], (3, 6], etc.).

```python
import pandas as pd

# Sample data: monthly expenses in dollars
expenses = pd.Series([1000, 250, 300, 600, 750, 800, 1200, 2000])

# Define bins: 0-300 = Low, 300-800 = Medium, 800-2000 = High
# 'bins' is a list of the cut points; 'labels' give names to each bin
binned_expenses = pd.cut(expenses,
                         bins=[0, 300, 800, 2000],
                        #  labels=['Low', 'Medium', 'High']
                         )

print("Expenses:")
print(expenses)
print("\nBinned Expenses (using cut):")
print(binned_expenses)

# You can also see how many items fall into each bin
print("\nCount of items in each bin:")
print(binned_expenses.value_counts())
```

#### .qcut() method

The qcut() is similar to cut(), but instead of defining fixed bin edges yourself, qcut() divides the data into bins based on quantiles (i.e., each bin has roughly the same number of data points).

Pro tip: Adjust the q parameter to create the number of quantile-based bins you want (e.g., q=4 for quartiles).

```python
import pandas as pd

# Same monthly expenses data
expenses = pd.Series([1000, 250, 300, 600, 750, 800, 1200, 2000])

# We'll split the data into 3 quantile-based bins
quantile_binned_expenses = pd.qcut(expenses, q=3, labels=['Low', 'Medium', 'High'])

print("Expenses:")
print(expenses)
print("\nQuantile-Binned Expenses (using qcut):")
print(quantile_binned_expenses)

# Again, see how many items in each quantile-based bin
print("\nCount of items in each bin:")
print(quantile_binned_expenses.value_counts())
```

### Q9: pivot_table() method

> Q9: If you had to teach a new user about the pivot_table() method, what would you tell them?

The pivot_table() method in pandas is used to create a spreadsheet-style pivot table, allowing you to summarize and reorganize data. It aggregates data using a function (e.g., mean, sum), with options for indexing by rows (index) and columns (columns).

```python
import pandas as pd

# Sample data: employee sales in different stores
data = {
    'Employee': ['Alice', 'Alice', 'Charlie', 'Charlie', 'Charlie', 'Charlie'],
    'Store': ['North', 'North', 'West', 'East', 'North', 'West'],
    'Sales': [200, 150, 300, 400, 250, 500]
}

df = pd.DataFrame(data)
# Create a pivot table showing total sales by Employee and Store
pivot = pd.pivot_table(
    df,
    index='Employee',                   # Row grouping
    columns='Store',                    # Column grouping
    values='Sales',                     # What we’re aggregating
    aggfunc=['count', 'sum', 'mean'],   # How we aggregate
    fill_value=0                        # Fill missing values with 0
)

print(pivot)
```

### Q10: Python for Data Analysis

> Q10: When is Python good for data analysis? When is it not?

#### When Python is Not the Best Option

Python has its limitations and may not be the best choice in certain situations:

1. Real-Time or High-Frequency Data Analysis  
	•	Python is not the fastest language for real-time data processing or when ultra-low latency is required.  
	•	Alternate Tools: Use C++, Java, or languages like Go for faster execution.  
	•	Example: High-frequency trading systems that need microsecond-level latency.  

2. Data Analysis in Spreadsheet-Heavy Workflows  
	•	While Python supports Excel manipulation (openpyxl, xlwings), some tasks are faster and more intuitive in spreadsheet software.  
	•	Alternate Tools: Excel or Google Sheets.  
	•	Example: A small business performing quick financial modeling or one-off data summaries.  

3. Large-Scale Business Intelligence Dashboards  
	•	While Python can create visualizations, it lacks the interactivity and scalability of dedicated BI tools.  
	•	Alternate Tools: Tableau, Power BI, or QlikView.  
	•	Example: A corporate dashboard summarizing financial KPIs for executives.  

4. Complex Statistical Analysis  
	•	While Python has libraries like statsmodels and scipy, R is often preferred for advanced statistical modeling.  
	•	Alternate Tools: R, SAS, or Stata.  
	•	Example: A public health organization conducting epidemiological studies with R for survival analysis.  

5. Handling Extremely Large Datasets  
	•	Python may struggle with in-memory operations on large datasets.  
	•	Alternate Tools: Apache Spark, Hadoop, or SQL-based solutions.  
	•	Example: A social media company analyzing terabytes of user activity logs for pattern recognition.  

#### When Python is Good for Data Analysis

Python is well-suited for data analysis in various scenarios due to its flexibility, extensive libraries, and community support:  

1. Exploratory Data Analysis (EDA)  
	•	Python excels at handling datasets for initial exploration.  
	•	Libraries: pandas, matplotlib, seaborn, and plotly are commonly used for cleaning, exploring, and visualizing data.  
	•	Example: A marketing team analyzing customer behavior data to identify trends using Jupyter Notebooks for visualization and insights.  

2. Machine Learning Workflows  
	•	Python is ideal for developing and testing machine learning models.  
	•	Libraries: scikit-learn, TensorFlow, PyTorch, and XGBoost.  
	•	Example: A retail company predicting customer churn using logistic regression or decision trees implemented in Python.  

3. Big Data Processing  
	•	Python, combined with libraries like PySpark or Dask, can manage large datasets for analysis.  
	•	Example: An e-commerce platform analyzing clickstream data to optimize website layout for user engagement.  

4. Automating Repetitive Data Tasks  
	•	Python scripts can automate tasks like data extraction, transformation, and loading (ETL).  
	•	Libraries: Airflow for workflows, requests for API interaction.  
	•	Example: Automating the daily data extraction from APIs for dashboard updates in a logistics company.  

5. Integration with Other Tools  
	•	Python integrates well with databases (e.g., SQLAlchemy for SQL databases), cloud platforms, and visualization tools (e.g., Tableau or Power BI).  
	•	Example: A fintech company querying databases with Python and sending processed data to Tableau dashboards.  

### Q11: Performance Considerations

> Q11: What are some performance considerations using Pandas? How do you optimize your code for speed?

*	**Avoid Row-by-Row Operations:** Row-by-row processing (e.g., loops or apply()) is slow and inefficient for large datasets.   
*	**Leverage Vectorized Operations:** Use built-in functions that operate on entire columns at once, leveraging Pandas’ C-optimized backend.   
*	**Optimize Data Types:** Reduce memory usage by using efficient data types (e.g., int8, float32) instead of defaults like int64 or float64.  
*	**Minimize Chained Operations:** Avoid chaining multiple operations without assigning intermediate results to variables.  
*	**Use Indexing Efficiently:** Use .loc[] or .iloc[] instead of chained indexing to avoid redundant DataFrame operations.  
*	**Consider Alternative Libraries for Large Datasets:** 
	-	Use Dask or modin for parallelized operations.  
	-	Use NumPy for simpler data structures if advanced features are unnecessary.  

### Q12: Handling Messy Data

> Q12: Describe a situation where you've had to clean messy data. What were the challenges and how did you handle them?

#### Situation:

I worked on a project where I had to analyze customer feedback data from multiple sources, including survey results, emails, and social media comments. The dataset was messy due to inconsistent formats, missing values, duplicate entries, and a mix of structured and unstructured data.

#### Challenges:

1.	Inconsistent Formats:  
•	Date fields were in multiple formats (e.g., MM/DD/YYYY, DD-MM-YYYY).  
•	Text fields had inconsistent capitalization and included emojis or special characters.  
2.	Missing Values:  
•	Some fields, such as customer age and location, were partially missing.  
3.	Duplicates:  
•	Multiple entries for the same customer appeared due to repeated survey submissions.  
4.	Unstructured Data:  
•	Social media comments and emails contained free text that needed to be processed and analyzed.  

#### Actions:

1. Standardizing Data Formats:
	•	Used Pandas to standardize date formats and clean text data.
2. Handling Missing Values:
	•	For numeric fields like age, used mean or median imputation.
    •	For categorical fields like location, used the most frequent value or created a placeholder ("Unknown").
3. Removing Duplicates:
	•	Identified and dropped duplicate rows based on a unique customer identifier.
4. Processing Unstructured Data:
	•	Used Natural Language Processing (NLP) techniques to clean and extract insights from text data:
	•	Removed stop words.
	•	Tokenized text for analysis.

#### Results:

1.	Improved Data Quality: Cleaned and standardized dataset was ready for analysis.
2.	Actionable Insights: Generated customer sentiment analysis and identified key trends.
3.	Scalable Workflow: Created a reusable data-cleaning pipeline for future projects.

**Key Takeaways:**  
	•	Cleaning messy data requires identifying issues, applying systematic fixes, and ensuring consistency.  
	•	Automating processes with libraries like Pandas and NLP tools helps streamline data cleaning.  
	•	A well-cleaned dataset leads to more reliable analysis and actionable insights.  