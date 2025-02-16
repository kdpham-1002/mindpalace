---
layout: post
title: IBM Mock Coding Interview
date: 2025-02-12 15:14 -0800
description: IBM mock coding Hackerrank OA
author: khoa_pham
categories: [Programming Hub, Career Preps]
tags: [python, SQL, interview preps, coding]
pin: false
math: true
mermaid: true
toc: true
comments: true
---

### Q1: Count the Employees   
The data for the number employed at several famous IT companies is maintained in the `COMPANY` table. Write a query to print the `ID`s of the companies that have more than **10,000** employees, in ascending order of `ID`.

**Input Format:**   

| Name        | Type    | Description |
|-------------|---------|-------------|
| `ID`        | Integer | A company ID in the inclusive range **[1,1000]**. This is the primary key. |
| `NAME`      | String  | A company name. This field contains between **1 and 100** characters (inclusive). |
| `EMPLOYEES` | Integer | The total number of employees in the company. |

**Output Format:**  
The result should contain the `ID`s of all the companies that have more than **10,000** employees, in ascending order in the following format:
```
COMPANY.ID
```

**Sample Input:**    

| ID | NAME       | EMPLOYEES |  
|----|------------|-----------|  
| 1  | Adobe      | 28085     |
| 2  | Flipkart   | 35543     |
| 3  | Amazon     | 1089      |
| 4  | Paytm      | 9982      |
| 5  | BookMyShow | 5589      |
| 6  | Oracle     | 4003      |
| 7  | NIIT       | 57782     |
| 8  | Samsung    | 2000      |
| 9  | TCS        | 10046     |
| 10 | Wipro      | 3500      |  

**Sample Output:**
```
1
2
7
9
```

**Explanation:**  
**Adobe, Flipkart, NIIT, and TCS** have greater than **10,000** employees, so their `ID`s are printed.


#### Q1 Solution:

```sql
SELECT ID
FROM COMPANY
WHERE EMPLOYEES > 10000
ORDER BY ID;
```




### Q2: FizzBuzz   
Given a number ***n***, for each integer ***i*** in the range from ***1 to n*** inclusive, print one value per line as follows:  
- If ***i*** is a multiple of both ***3*** and ***5***, print ***FizzBuzz***.
- If ***i*** is a multiple of ***3*** (but not ***5***), print ***Fizz***.
- If ***i*** is a multiple of ***5*** (but not ***3***), print ***Buzz***.
- If ***i*** is not a multiple of ***3*** or ***5***, print the value of ***i***.

**Function Description:**  
Complete the function *fizzBuzz* in the editor below.
*fizzBuzz* has the following parameter(s):  
   - `int n`: upper limit of values to test (inclusive)

***Returns:*** NONE  
***Prints:*** The function must print the appropriate response for each value ***i*** in the set `{1, 2, ... n}` in ascending order, each on a separate line.



**Constraints:**  
$$ 0 < n < 2 \times 10^5 $$

**Input Format for Custom Testing:**   
Input from stdin will be processed as follows and passed to the function.   
The single integer **n**, the limit of the range to test: `[1, 2, ... n]`.

**Sample Case 0**   
**Sample Input:**  
```
STDIN   Function
-----   --------
15  ->  n = 15
```

**Sample Output:**  
```
1
2
Fizz
4
Buzz
Fizz
7
8
Fizz
Buzz
11
Fizz
13
14
FizzBuzz
```

**Explanation:**  
- The numbers **3, 6, 9, and 12** are multiples of **3** (but not **5**), so print **Fizz** on those lines.
- The numbers **5 and 10** are multiples of **5** (but not **3**), so print **Buzz** on those lines.
- The number **15** is a multiple of both **3 and 5**, so print **FizzBuzz** on that line.
- None of the other values is a multiple of either **3 or 5**, so print the value of **i** on those lines.


#### Q2 Solution:

```python
# Complete the 'fizzBuzz' function below.
# The function accepts INTEGER n as parameter.
def fizzBuzz(n):
    # Write your code here
    for i in range(1, n + 1):
        if i % 3 == 0 and i % 5 == 0:
            print("FizzBuzz")
        elif i % 3 == 0:
            print("Fizz")
        elif i % 5 == 0:
            print("Buzz")
        else:
            print(i)

if __name__ == '__main__':
    n = int(input().strip())

    fizzBuzz(n)
```