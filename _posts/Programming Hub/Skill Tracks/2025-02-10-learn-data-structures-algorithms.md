---
layout: post
title: Learn Data Structures & Algorithms
date: 2025-02-10 20:50 -0800
description: Data Structures & Algorithms Preparation in Python
author: khoa_pham
categories: [Programming Hub, Skill Tracks]
tags: [python, interview preps, coding]
pin: false
math: true
mermaid: true
toc: true
comments: true
---

## 1. Lists/Arrays

#### 1_Remove duplicates from a list
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
print(remove_duplicates(input)) # [1, 2, 3, 4, 5]
```



#### 2_Find the Maximum Product of Two Numbers in a List
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
print(max_product(input))  # 20
```



#### 3_Check if List is Sorted
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



#### 4_Rotate a List by K Positions
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



#### 5_Find the Missing Number in a Sequence
```python
# Given a list of numbers from 1 to n with one number missing, find the missing number.
"""
1. Calculate the sum of the numbers from 1 to n using the formula n * (n + 1) // 2.
2. Subtract the sum of the list from the sum of the numbers from 1 to n to find the missing number.
"""
def find_missing_number(arr):
    n = len(arr) + 1
    experted_sum = n * (n + 1) // 2
    actual_sum = sum(arr)
    return experted_sum - actual_sum

input = [1, 2, 3, 5]
print(find_missing_number(input))  # Output: 4
```



#### 6_Reverse a List
```python
# Reverse a list in-place.
"""
1. Use two pointers, one at the start and the other at the end of the list.
2. Swap the elements at the two pointers and move the pointers towards the center.
3. Continue swapping until the two pointers meet.
"""
def reverse_list(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        arr[left], arr[right] = arr[right], arr[left]
        left += 1
        right -= 1
    return arr

input = [1, 2, 3, 4, 5]
print(reverse_list(input))  # Output: [5, 4, 3, 2, 1]
```



#### 7_Find all Pairs with a Given Sum
```python
# Given a list of numbers and a target sum, find all unique pairs whose sum equals the target.
"""
1. Use a set to store the numbers seen so far.
2. Iterate through the list and check if (target - current_number) is in the set.
3. If it is, add the pair (current_number, target - current_number) to the result set.
"""

def find_pairs(arr, target):
    seen = set()
    result = []
    for num in arr:
        diff = target - num
        if diff in seen:
            result.append((diff, num))
        seen.add(num)
    return result

print(find_pairs([2, 4, 3, 7, 1, 5, 9], 6))  # Output: [(4, 2), (1, 5)]

```



#### 8_Find the Second Largest Number in a List
```python
# Find the second largest number in a list without sorting.
"""
1. Use two variables: largest and second_largest.
2. Iterate through the list and update these variables accordingly.
"""
def second_largest(arr):
    largest = second = float('-inf')
    for num in arr:
        if num > largest:
            second = largest
            largest = num
        elif num > second and num != largest:
            second = num

    # return second if second != float('-inf') else None
    if second != float('-inf'):
        return second
    else:
        return None

print(second_largest([1, 2, 3, 4, 5]))  # Output: 4
print(second_largest([10, 20, 20, 15, 10]))  # Output: 15
print(second_largest([50, 10, 40, 20, 30]))  # Output: 40
print(second_largest([5, 5, 5, 5]))  # Output: None
print(second_largest([-10, -20, -30, -5]))  # Output: -10
```


#### 9_Move All Zeros to the End
```python
# Move all zeroes to the end of the list while maintaining the relative order of non-zero elements.
"""
1. Use two pointers: one to iterate through the list and another to keep track of the position to place the next non-zero element.
2. Swap the non-zero element with the zero element.
"""

def move_zeros(arr):
    insert_pos = 0
    for num in arr:
        if num != 0:
            arr[insert_pos] = num
            insert_pos += 1
    while insert_pos < len(arr):
        arr[insert_pos] = 0
        insert_pos += 1
    return arr

# def move_zeros(arr):
#     zero_pos = 0
#     for i in range(len(arr)):
#         if arr[i] != 0:
#             arr[i], arr[zero_pos] = arr[zero_pos], arr[i]
#             zero_pos += 1
#     return arr

print(move_zeros([0, 1, 0, 3, 12]))  # Output: [1, 3, 12, 0, 0]
```



#### 10_Find the Intersection of Two Lists
```python
# Find common elements in two lists.
"""
1. Use a set to store the elements of the first list.
2. Iterate through the second list and check if the element is in the set.
3. If it is, add it to the result set.
"""
def find_intersection(arr1, arr2):
    set1 = set(arr1)  
    # return [num for num in arr2 if num in set1]
    result = []
    for num in arr2:
        if num in set1: 
            result.append(num)
    return result

print(find_intersection([1, 2, 3, 4], [2, 4, 6, 8]))  # Output: [2, 4]
```



## 2. Hash Tables

#### 1_Count Frequency of Elements
```python
# Count how many times each number appears in a list.
"""
1. Create an empty dictionary to store the frequency of each number.
2. Iterate through the list and update the dictionary accordingly.
"""
def count_frequency(arr):
    freq = {}
    for num in arr:
        freq[num] = freq.get(num, 0) + 1
    return freq

print(count_frequency([2, 3, 2, 4, 3, 2, 5])) # Output: {2: 3, 3: 2, 4: 1, 5: 1}
```



#### 2_Find the First Non-Repeating Element
```python
# Find the first character that appears only once in a given string.
"""
1. Create an empty dictionary to store the frequency of each character.
2. Iterate through the string and update the dictionary accordingly.
3. Iterate through the string again and return the first character with a frequency of 1.
"""
def first_unique_char(s):
    freq = {}
    for char in s:
        freq[char] = freq.get(char, 0) + 1
    for char in s:
        if freq[char] == 1:
            return char
    return None

print(first_unique_char("swiss"))  # Output: "w"
```



#### 3_Two Sum Problem
```python
# Given a list of numbers and a target sum, find two numbers that add up to the target.
"""
1. Create an empty dictionary to store the index of each number.
2. Iterate through the list and check if the difference between the target and the current number is in the dictionary.
3. If it is, return the indices of the two numbers.
"""
def two_sum(arr, target):
    num_map = {}
    for i, num in enumerate(arr):
        diff = target - num
        if diff in num_map:
            return [num_map[diff], i]
        num_map[num] = i
    return None
    
print(two_sum([2, 6, 11, 7], 9))  # Output: (0, 1)
```



#### 4_Find Intersection of Two Lists
```python
# Find common elements in two lists.
"""
1. Create a set from the first list.
2. Iterate through the second list and check if the element is in the set.
3. If it is, add it to the result set.
"""
def list_intersection(arr1, arr2):
    set1 = set(arr1)
    return [num for num in arr2 if num in set1]
    # result = []
    # for num in arr2: 
    #     if num in set1: 
    #         result.append(num)
    # return result
print(list_intersection([1, 2, 3, 4], [2, 4, 6, 8]))  # Output: [2, 4]
```


#### 5_Find Duplicates in a List
```python
# Find all duplicate elements in a list.
"""
1. Create an empty set to store the unique elements.
2. Create an empty list to store the duplicate elements.
3. Iterate through the list and check if the element is in the set.
4. If it is, add it to the duplicate list; otherwise, add it to the set.
"""
def find_duplicates(arr):
    seen = set()
    duplicates = []
    for num in arr:
        if num in seen:
            duplicates.append(num)
        else:
            seen.add(num)
    return duplicates

# def find_duplicates(arr):
#     freq = {}
#     duplicates = []
#     for num in arr:
#         freq[num] = freq.get(num, 0) + 1
#     for num, count in freq.items():
#         if count > 1:
#             duplicates.append(num)
#     return duplicates

print(find_duplicates([4, 2, 7, 4, 8, 2]))  # Output: [4, 2]
```



## 3. Stacks & Queues

### Stacks (LIFO - Last In, First Out)
#### 1_Reversing a String
```python
# Reverse a string using a stack.
"""
1. Create an empty stack to store characters.
2. Push each character of the string onto the stack.
3. Pop each character from the stack to get the reversed string.
"""
def reverse_string(s):
    stack = []
    for char in s:
        stack.append(char)
    reversed_str = ""
    while stack:
        reversed_str += stack.pop()
    return reversed_str

print(reverse_string("hello"))  # Output: "olleh"
```

#### 2_Check Balanced Parentheses
```python
# Given a string containing only (), {}, and [], determine if it is balanced.
"""
1. Create an empty stack to store opening parentheses.
2. Iterate through the string and push opening parentheses onto the stack.
3. If a closing parenthesis is encountered, pop the stack and check if it matches the closing parenthesis.
4. If the stack is empty at the end, the parentheses are balanced.
"""
def is_balanced(s):
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}
    
    for char in s:
        if char in pairs.values():  
            stack.append(char)  
        elif char in pairs.keys():  
            if not stack or stack.pop() != pairs[char]:  
                return False  
    return len(stack) == 0  
    # return not stack


print(is_balanced("{[()]}"))  # Output: True
print(is_balanced("{[(])}"))  # Output: False
```

### Queues (FIFO - First In, First Out)
#### 3_Implement a Queue
