---
layout: post
title: Learn Data Structures & Algorithms
date: 2025-02-10 20:50 -0800
description: Data Structures & Algorithms Preparation for Online Coding Interviews
author: khoa_pham
categories: [Coding Workspace, Interview Preps]
tags: [python, interview, coding]
pin: false
math: true
mermaid: true
toc: true
comments: true
---

## 1. Lists (Arrays)
### 1. Remove duplicates from a list
```python
# Given a list of integers, remove all duplicate elements and return the unique values.
"""
1. Remove duplicates but keep the order
2. Since `set()` does not keep the order, use `dict.fromkeys(arr)` to keep the order.
3. Convert the dictionary keys back to a list using `list()`.
"""
def remove_duplicates(arr):
    return list(dict.fromkeys(arr))

input = [1, 2, 2, 3, 4, 4, 5]
print(input)
print(remove_duplicates(input)) # [1, 2, 3, 4, 5]
```

### 2. Find the Maxiumum Product of Two Numbers in a List
```python
# Find the maximum product that can be obtained by multiplying two numbers in a list.
"""
1. Sort the list in ascending order.
2. The maximum product is either the product of the two largest numbers or the two smallest numbers.
3. Return the maximum of the two products -> max(20, 12) = 20
"""
def max_product(arr):
    arr.sort()
    print(arr)
    return max(arr[-1] * arr[-2], arr[0] * arr[1])

input = [3, 5, -2, -6, 4]
print(input)
print(max_product(input))  # 30
```

### 3. Check if List is Sorted
```python
# Check whether a list is sorted in ascending order.

## Approach 1: Compare each element with the next element.
"""
1. Iterate through the list and compare each element with the next element.
2. If any element is greater than the next element, return False.
3. If the loop completes without finding any such element, return True.
# O(n) because it iterates through the list once.
"""
def is_sorted(lst):
    for i in range(len(lst) - 1):
        if lst[i] > lst[i + 1]:
            return False
    return True

## Approach 2: Compare the list with the sorted list.
"""
1. Compare the list with the sorted list.
2. If the list is sorted, the two lists will be equal, return True.
# O(nlogn) because sorted() creates a new list and sorts it.
"""
def is_sorted(arr):
    return arr == sorted(arr)

input1 = [1, 2, 3, 4, 5]
input2 = [5, 3, 2, 1]
print(is_sorted(input1))  # Output: True
print(is_sorted(input2))  # Output: False
```
