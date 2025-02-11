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

## Lists (Arrays)

### Remove duplicates from a list
```python
# Given a list of integers, remove all duplicate elements and return the unique values.
input = [1, 2, 2, 3, 4, 4, 5]
print(input)

def remove_duplicates(arr):
    return list(dict.fromkeys(arr))
print(remove_duplicates(input))
```

### Find the Maxiumum Product of Two Numbers in a List
```python
# Find the maximum product that can be obtained by multiplying two numbers in a list.
input = [3, 5, -2, -6, 4]
print(input)

def max_product(arr):
    arr.sort()
    print(arr)
    return max(arr[-1] * arr[-2], arr[0] * arr[1])
print(max_product(input))  
```
