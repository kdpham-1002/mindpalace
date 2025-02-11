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

### 4. Rotate a List by K Positions
```python
# Given a list and an integer K, rotate the list rightward by K positions.
"""
1. Calculate the effective rotation by taking k modulo the length of the list.
2. Slice the list into two parts: the last k elements and the rest of the list.
3. Concatenate the two parts to get the rotated list.
"""
def rotate_list(arr, k):
    k = k % len(arr)
    return arr[-k:] + arr[:-k]

input = [1, 2, 3, 4, 5]
print(rotate_list(input, 2))  # Output: [4, 5, 1, 2, 3]
print(rotate_list(input, 7))  # Output: [4, 5, 1, 2, 3]
```

### 5. Find the Missing Number in a Sequence
```python
# Given a list of numbers from 1 to n with one number missing, find the missing number.
"""
1. Calculate the sum of the numbers from 1 to n using the formula n * (n + 1) // 2.
2. Subtract the sum of the list from the sum of the numbers from 1 to n to find the missing number.
"""
def find_missing_number(arr):
    n = len(arr) + 1
    return (n*(n+1) // 2) - sum(arr)

input = [1, 2, 3, 5]
print(input)
print(find_missing_number(input))  # Output: 4
```

### 6. Reverse a List
```python
# Reverse a list in-place.
"""
1. Use two pointers, one at the start and the other at the end of the list.
2. Swap the elements at the two pointers and move the pointers towards the center.
3. Continue swapping until the two pointers meet.
"""
