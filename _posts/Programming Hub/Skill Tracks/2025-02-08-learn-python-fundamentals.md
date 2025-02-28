---
layout: post
title: Learn Python Fundamentals
date: 2025-02-08 19:10 -0800
description: Python Fundamentals and Pandas Basics
author: khoa_pham
categories: [Programming Hub, Skill Tracks]
tags: [python, interview preps, coding]
pin: false
math: true
mermaid: true
toc: true
comments: true
---

## 1) Intro to Python
### Python basics
* Calculations
* Variables and types (int, float, str, bool)

### Lists
* Subsetting, slicing; list of lists
* Replace, extend, delete
* Make copy of list 

### Functions and Packages
* Familiar functions
    * help(pow), help(max),…
    * type(), len(), int(), str(), sorted(iterable =, reverse =)
* String methods
    * .upper(), .count(), .index()
    * .append(), .remove(), .reverse()
* Import packages

### NumPy
* NumPy Fundamentals

### Matplotlib
* Histogram
* Line Plot
* Scatter Plot
* Customizations (size, color, grid…)

### Dictionaries
* Create, access dictionaries
* Add, delete; dictionary in dictionaries

### Pandas
* pd.read_csv('cars.csv', index_col = 0)
* Pandas Series vs DataFrame; print columns
* Indexing; loc vs iloc

### Logic, Control Flow, Filtering and Loops
* Comparison, Boolean Operators
* If-Elif-Else
* While loop
* For loop
    * Loop using enumerate()
    * Loop over list of lists
    * Loop over dictionary
    * Loop over NumPy array
    * Loop over DataFrame
    * Add column; .apply(str.upper)



## 2) Data Manipulation & Joining Data with Pandas
### Sorting and Subsetting
* .head(), .info(), .shape, .describe()
* .values, .columns, .index
* Sorting, subsetting, adding
    * .sort_values()
    * .isin()


### Aggregating Dataframes
* .mean(), .median(), .max(), .min()
* .agg([iqr, np.median])
* .cumsum(), .cummax()
* .drop_duplicates(subset=["store", "department"])
* .value_counts(sort=True, normalize=True)
    * Adding ‘normalize’ argument to get proportion
* .groupby("type")[["unemployment", "fuel_pricel"]].agg([np.min, np.mean])
* .pivot_table()
    * filling in any missing values with 0 and summing all rows and columns

```python
sales.pivot_table(values="weekly_sales", index="department", 
                    columns="type", fill_value=0, margins=True)
```

### Slicing and Indexing
* Set and reset index
* Subsetting with.loc[]
* Multi-level indexes
* Sorting by index values
* Slicing index values, in both directions, time series, by row/column number
* Subsetting pivot tables and calculating

### Handling and Visualizing Data
* Finding, removing, replacing missing values
* List of dictionaries; Dictionary of lists
* CSV to df; df to CSV


### Data Merging
### Different Join Types
### Advanced Merging and Concatenation
### Merging Ordered and Time Series Data


## 3) Functions in Python
### User-defined Functions
#### Defining a Function
```python
def square():
    new_value = 4 ** 2
    print(new_value)

square()  # Output: 16
```

#### Function Parameters
```python
def square(value):
    new_value = value ** 2
    print(new_value)

square(4)  # Output: 16
square(5)  # Output: 25
```

#### Returning Values
```python
def square(value):
    new_value = value ** 2
    return new_value

num = square(4)
print(num)  # Output: 16
```

#### Using Docstrings
```python
def square(value):
    """Returns the square of a value."""
    return value ** 2
```

### Arguments and Scope
#### Scope
```python
new_val = 10  # Global variable

def square(value):
    new_val = value ** 2  # Local variable
    return new_val

print(square(3))  # Output: 9
print(new_val)    # Output: 10  (global value remains unchanged)
```

#### `global` Keyword
```python
new_val = 10

def square(value):
    global new_val
    new_val = new_val ** 2  # Modifies global variable
    return new_val

print(square(3))  # Output: 100
print(new_val)    # Output: 100
```

#### Nested Functions
```python
def mod2plus5(x1, x2, x3):
    """Returns remainder +5 for three numbers."""
    
    def inner(x):
        return x % 2 + 5  # Nested function
    
    return inner(x1), inner(x2), inner(x3)

print(mod2plus5(1, 2, 3))  # Output: (6, 5, 6)
```

```python
def raise_val(n):
    """Returns a function that raises numbers to power n."""
    def inner(x):
        return x ** n
    return inner

square = raise_val(2)
cube = raise_val(3)

print(square(2), cube(4))  # Output: 4, 64
```

`nonlocal` keyword: modifies enclosing variables.

```python
def outer():
    n = 1
    def inner():
        nonlocal n
        n = 2
        print(n)
    inner()
    print(n)

outer()  # Output: 2, 2
```

#### Default Arguments
```python
def power(number, pow=1):
    """Raise number to the power of pow."""
    return number ** pow

print(power(9, 2))  # Output: 81
print(power(9))     # Output: 9 (default pow=1)
```

#### Flexible Arguments (*args & **kwargs)
```python
def add_all(*args):
    """Sum all values passed as arguments."""
    return sum(args)

print(add_all(1, 2, 3, 4))  # Output: 10
```

```python
def print_all(**kwargs):
    """Print key-value pairs."""
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_all(name="John", age=30)

# Output:
# name: John
# age: 30
```

### Lambda Functions
```python
raise_to_power = lambda x, y: x ** y
print(raise_to_power(2, 3))  # Output: 8
```

#### `map()`, `filter()`, and `reduce()`

`map()` function: applies a function to all items in an iterable.
```python
nums = [48, 6, 9, 21, 1]
square_all = list(map(lambda num: num ** 2, nums))
print(square_all)  # Output: [2304, 36, 81, 441, 1]
```

`filter()` function: filters items in an iterable based on a function.
```python
nums = [48, 6, 9, 21, 1]
even_nums = list(filter(lambda num: num % 2 == 0, nums))
print(even_nums)  # Output: [48, 6]
```

`reduce()` function: applies a rolling computation to sequential pairs of values in an iterable.
```python
from functools import reduce
nums = [48, 6, 9, 21, 1]
sum_all = reduce(lambda x, y: x + y, nums)
print(sum_all)  # Output: 85
```

### Error-handling
#### `try-except` 
```python
def sqrt(x):
    """Returns square root of a number."""
    try:
        return x ** 0.5
    except TypeError:
        print('x must be an int or float')

print(sqrt(4))     # Output: 2.0
print(sqrt("hi"))  # Output: x must be an int or float
```

#### `raise`
```python
def sqrt(x):
    if x < 0:
        raise ValueError("x must be non-negative")
    return x ** 0.5

print(sqrt(-2))  # ValueError: x must be non-negative
```

#### `finally`
```python
def divide(x, y):
    try:
        return x / y
    except ZeroDivisionError:
        print("Cannot divide by zero")
    finally:
        print("Execution completed")
print(divide(4, 2))  # Output: 2.0, Execution completed
print(divide(4, 0))  # Output: Cannot divide by zero, Execution completed
```

## 4) Python Toolbox
### Iterators
#### Iterating with `for` loop
```python
employees = ['Nick', 'Lore', 'Hugo']
for employee in employees:
    print(employee)
# Output:
# Nick
# Lore
# Hugo
```

```python
for letter in 'DataCamp':
    print(letter)
# Output: D a t a C a m p
```

```python
for i in range(5):
    print(i)
# Output: 0 1 2 3 4
```

#### Iterators vs Iterables
* Iterable: object that can be looped over (list, str, dict, etc.)
* Iterator: object with __next__() method to get next value
* iter(): converts an iterable into an iterator.
* next(): gets the next value from an iterator.

```python
numbers = [1, 2, 3]
it = iter(numbers)
print(next(it))  # Output: 1
print(next(it))  # Output: 2
print(next(it))  # Output: 3
print(next(it))  # Error: StopIteration
```

```python
word = 'Data'
it = iter(word)
print(*it) # Output: D a t a
```

#### Iterating over dictionaries
```python
employees = {'Nick': 100, 'Lore': 200, 'Hugo': 300}
for key, value in employees.items():
    print(key, value)

# Output:
# Nick 100
# Lore 200
# Hugo 300
```

#### Using `enumerate()`
```python
employees = ['Nick', 'Lore', 'Hugo']
for index, employee in enumerate(employees, start=10):
    print(index, employee)

# Output:
# 10 Nick
# 11 Lore
# 12 Hugo
```

#### Using `zip()`
```python
names = ['Nick', 'Lore', 'Hugo']
salaries = [100, 200, 300]
for name, salary in zip(names, salaries):
    print(name, salary)

# Output:
# Nick 100
# Lore 200
# Hugo 300
```

```python
avengers = ['hawkeye', 'iron man', 'thor', 'quicksilver']
names = ['barton', 'stark', 'odinson', 'maximoff']
z = zip(avengers, names)
print(*z)

# Output:
# ('hawkeye', 'barton') ('iron man', 'stark') ('thor', 'odinson') ('quicksilver', 'maximoff')
```

#### Using `reversed()`
```python
numbers = [1, 2, 3]
for number in reversed(numbers):
    print(number)

# Output: 3 2 1
```

#### Using `sorted()`
```python
numbers = [3, 1, 2]
for number in sorted(numbers):
    print(number)

# Output: 1 2 3
```

#### Loading Large Files with `chunksize`

Avoids memory overload when processing large datasets.

```python
import pandas as pd
total = 0
for chunk in pd.read_csv('data.csv', chunksize=1000):
    total += sum(chunk['x'])
print(total)
```


### List Comprehensions
#### Basic List Comprehension

Syntax: `[expression for item in iterable]`

```python
numbers = [1, 2, 3]
squared = [number ** 2 for number in numbers]
print(squared)  # Output: [1, 4, 9]
```

```python
result = [num for num in range(11)]
print(result) # Output: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

#### Nested List Comprehension
```python
pairs = [(num1, num2) for num1 in range(2) for num2 in range(6, 8)]
print(pairs)
# Output: [(0, 6), (0, 7), (1, 6), (1, 7)]
```

#### List Comprehension with `if-else`
```python
numbers = [1, 2, 3, 4, 5]
squared = [number ** 2 for number in numbers if number % 2 == 0]
print(squared)  # Output: [4, 16]
```

```python
numbers = [1, 2, 3, 4, 5]
squared = [number ** 2 if number % 2 == 0 else 0 for number in numbers]
print(squared)  # Output: [0, 4, 0, 16, 0]
```

#### Dictionary Comprehension
```python
pos_neg = {num: -num for num in range(6)}
print(pos_neg)
# Output: {0: 0, 1: -1, 2: -2, 3: -3, 4: -4, 5: -5}
```

```python
numbers = [1, 2, 3, 4, 5]
squared = {number: number ** 2 for number in numbers if number % 2 == 0}
print(squared)  # Output: {2: 4, 4: 16}
```

### Generators   
Similar to list comprehensions, but generates values lazily (one at a time).
* Lazy Evaluation: Values are produced only when needed, saving memory.

```python
result = (num for num in range(6))
print(next(result))  # Output: 0
print(next(result))  # Output: 1
```

```python
even_nums = (num for num in range(10) if num % 2 == 0)
print(list(even_nums))
# Output: [0, 2, 4, 6, 8]
```

```python
# Creating Generator Functions with `yield`
# yield instead of return → Saves memory by generating values one at a time.

def num_sequence(n):
    """Generate values from 0 to n."""
    i = 0
    while i < n:
        yield i
        i += 1

result = num_sequence(5)
print(list(result))
# Output: [0, 1, 2, 3, 4]
```

> Even though you can only fetch one element at a time, you can iterate over the generator just like a list. Use case example: Reading a file line by line without loading everything into memory.




## 5) Importing Data

### Importing Flat Files (.csv, .txt)
#### Reading Text Files
```python
filename = 'huck_finn.txt'
with open(filename, mode='r') as file: # 'r' is to read, 'w' is to write
    text = file.read()
print(text[:500])  # Print first 500 characters
```

#### Reading numerical data with NumPy
```python
import numpy as np
data = np.loadtxt('MNIST.txt', delimiter=',', skiprows=1, usecols=[0, 1, 2])
print(data[:5])
```

#### Reading CSV Files with Pandas
```python
import pandas as pd
df = pd.read_csv('titanic.csv', sep=',', skiprows=1, usecols=['Name', 'Survived'])
print(df.head())
```

### Importing other file formats
#### Excel Files (.xlsx)
```python
df = pd.read_excel('urbanpop.xlsx', sheet_name='1960-1966')
print(df.head())
```

#### MATLAB Files (.mat)
```python
import scipy.io
mat = scipy.io.loadmat('workspace.mat')
print(mat.keys()) # List variables inside the file

# Data is stored as a dictionary
```

#### SAS Files (.sas7bdat) and Stata Files (.dta)
```python
df_sas = pd.read_sas('data.sas7bdat')
df_stata = pd.read_stata('data.dta')
```

#### HDF5 Files
```python
import h5py
file = h5py.File('data.hdf5', 'r')
print(list(file.keys()))  # List datasets inside the file

# Used for big data and scalable data storage
```


### Querying Databases with SQL
#### Connecting to a Database
```python
from sqlalchemy import create_engine

engine = create_engine('sqlite:///Northwind.sqlite')
connection = engine.connect()

print(engine.table_names())
```

#### Querying the Database
```python
import pandas as pd

df = pd.read_sql_query("SELECT * FROM Orders LIMIT 5", engine)
print(df.head())
```

```python
# Querying with a filter

df = pd.read_sql_query("SELECT * FROM Orders WHERE EmployeeID = 5", engine)
print(df.head())
```

```python
# Joining tables (INNER JOIN)

query = """
SELECT OrderID, CompanyName FROM Orders
INNER JOIN Customers ON Orders.CustomerID = Customers.CustomerID
"""
df = pd.read_sql_query(query, engine)
print(df.head())
```
