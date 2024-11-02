From: https://techinterviewhandbook.org

# Array

Note that because both arrays and strings are sequences (a string is an array of characters), most of the techniques here will apply to string problems.

### Sliding window
Master the sliding window technique that applies to many subarray/substring problems. In a sliding window, the two pointers usually move in the same direction will never overtake each other. This ensures that each value is only visited at most twice and the time complexity is still O(n). Examples: Longest Substring Without Repeating Characters, Minimum Size Subarray Sum, Minimum Window Substring

Template
```
1. Initialize two pointers, start and end, both at the 0th index.

2. Initialize any needed variables. For example:
   - max_sum for storing the maximum sum of subarray
   - current_sum for storing the sum of the current window
   - any other specific variables you need

3. Iterate over the array/string using the end pointer:

   while end < length_of_array:
       
       a. Add the current element to current_sum (or perform some operation)

       b. While the current window meets a certain condition (like current_sum exceeds a value, the window has more than a specific number of characters, etc.):
          i. Possibly update max_sum or any other variables
          ii. Remove the element at start pointer from current_sum (or perform the opposite operation)
          iii. Increment the start pointer to make the window smaller

       c. Increment the end pointer

4. After the loop, the answer could be in max_sum or any other variables you used.
```

Example
```swift
// Find the maximum sum subarray of size `k`:
func maxSumSubarray(_ arr: [Int], _ k: Int) -> Int {
    var start = 0
    var currentSum = 0
    var maxSum = Int.min

    for end in 0..<arr.count {
        currentSum += arr[end]

        if end - start + 1 == k {
            maxSum = max(maxSum, currentSum)
            currentSum -= arr[start]
            start += 1
        }
    }

    return maxSum
}
```

### Two pointers
Two pointers is a more general version of sliding window where the pointers can cross each other and can be on different arrays. Examples: Sort Colors, Palindromic Substrings

When you are given two arrays to process, it is common to have one index per array (pointer) to traverse/compare the both of them, incrementing one of the pointers when relevant. For example, we use this approach to merge two sorted arrays. Examples: Merge Sorted Array

Template
```
1. Initialize two pointers. Depending on the problem:
   - They can start at the beginning and end of the array (`start = 0, end = length_of_array - 1`), which is typical for sorted arrays.
   - Or they can start both at the beginning (`left = 0, right = 1`) or any other positions based on the requirement.

2. Use a loop (typically a while loop) to iterate while the pointers meet the criteria for traversal, e.g., `start < end`.

3. Inside the loop:
   a. Check the current elements at the two pointers.
   b. Based on the problem, decide how to adjust the pointers. Common decisions are:
      i. If the current elements meet some condition, process the current elements and then adjust the pointers (either moving `start` forward or `end` backward or both).
      ii. If the current elements don't meet the condition, adjust one of the pointers (or both) without processing the elements.

4. Continue until the loop ends.

5. The solution might be obtained during the loop's iterations, or after the loop based on the processed elements.
```

Example
```swift
// In a sorted array, find a pair of elements that sum up to a given target:
func twoSumSorted(_ arr: [Int], _ target: Int) -> (Int, Int)? {
    var start = 0
    var end = arr.count - 1

    while start < end {
        let currentSum = arr[start] + arr[end]

        if currentSum == target {
            return (arr[start], arr[end])
        } else if currentSum < target {
            start += 1
        } else {
            end -= 1
        }
    }

    return nil // No pair found
}
```

### Traversing from the right
Sometimes you can traverse the array starting from the right instead of the conventional approach of from the left. Examples: Daily Temperatures, Number of Visible People in a Queue

### Sorting the array
Is the array sorted or partially sorted? If it is, some form of binary search should be possible. This also usually means that the interviewer is looking for a solution that is faster than O(n).

Can you sort the array? Sometimes sorting the array first may significantly simplify the problem. Obviously this would not work if the order of array elements need to be preserved. Examples: Merge Intervals, Non-overlapping Intervals

### Precomputation
For questions where summation or multiplication of a subarray is involved, pre-computation using hashing or a prefix/suffix sum/product might be useful. Examples: Product of Array Except Self, Minimum Size Subarray Sum, LeetCode questions tagged "prefix-sum"

### Index as a hash key
If you are given a sequence and the interviewer asks for O(1) space, it might be possible to use the array itself as a hash table. For example, if the array only has values from 1 to N, where N is the length of the array, negate the value at that index (minus one) to indicate presence of that number. Examples: First Missing Positive, Daily Temperatures

### Traversing the array more than once
This might be obvious, but traversing the array twice/thrice (as long as fewer than n times) is still O(n). Sometimes traversing the array more than once can help you solve the problem while keeping the time complexity to O(n).

## String

### Counting characters
Often you will need to count the frequency of characters in a string. The most common way of doing that is by using a hash table/map in your language of choice. If your language has a built-in Counter class like Python, ask if you can use that instead.

If you need to keep a counter of characters, a common mistake is to say that the space complexity required for the counter is O(n). The space required for a counter of a string of latin characters is O(1) not O(n). This is because the upper bound is the range of characters, which is usually a fixed constant of 26. The input set is just lowercase Latin characters.

### Anagram
An anagram is word switch or word play. It is the result of rearranging the letters of a word or phrase to produce a new word or phrase, while using all the original letters only once. In interviews, usually we are only bothered with words without spaces in them.

To determine if two strings are anagrams, there are a few approaches:

- Sorting both strings should produce the same resulting string. This takes O(n.log(n)) time and O(log(n)) space.
- If we map each character to a prime number and we multiply each mapped number together, anagrams should have the same multiple (prime factor decomposition). This takes O(n) time and O(1) space. Examples: Group Anagram
- Frequency counting of characters will help to determine if two strings are anagrams. This also takes O(n) time and O(1) space.

### Palindrome
A palindrome is a word, phrase, number, or other sequence of characters which reads the same backward as forward, such as madam or racecar.

Here are ways to determine if a string is a palindrome:

- Reverse the string and it should be equal to itself.
- Have two pointers at the start and end of the string. Move the pointers inward till they meet. At every point in time, the characters at both pointers should match.
- The order of characters within the string matters, so hash tables are usually not helpful.

When a question is about counting the number of palindromes, a common trick is to have two pointers that move outward, away from the middle. Note that palindromes can be even or odd length. For each middle pivot position, you need to check it twice - once that includes the character and once without the character. This technique is used in Longest Palindromic Substring.

- For substrings, you can terminate early once there is no match
- For subsequences, use dynamic programming as there are overlapping subproblems


## Hash Table

- Describe an implementation of a least-used cache, and big-O notation of it.

Template
```
Initialize an LRU Cache with a given capacity:

1. Set cache capacity.
2. Create an empty hashmap that will hold key-value pairs (key -> node in doubly-linked list).
3. Create an empty doubly-linked list (nodes will have key-value pairs).

To GET a value from the cache:

1. If the key is not in the hashmap:
   a. Return "Not Found" or equivalent.
2. If the key is in the hashmap:
   a. Use the hashmap to get the node associated with that key from the doubly-linked list.
   b. Move the accessed node to the head of the doubly-linked list (indicating recent use).
   c. Return the value of the node.

To PUT a value in the cache:

1. If the key is already in the hashmap:
   a. Use the hashmap to get the node associated with that key from the doubly-linked list.
   b. Update the value of the node.
   c. Move the node to the head of the doubly-linked list.
2. If the key is not in the hashmap:
   a. Create a new node with the key-value pair.
   b. Add the node to the head of the doubly-linked list.
   c. Add the key and the node reference to the hashmap.
   d. If the size of the hashmap exceeds the cache capacity:
      i. Remove the node from the tail of the doubly-linked list (least recently used item).
      ii. Remove the corresponding key from the hashmap.

Helper Method - MoveToHead(node):

1. Remove the node from its current position in the doubly-linked list.
2. Add the node to the head of the doubly-linked list.
```

Example
```swift
class ListNode {
    var key: Int
    var value: Int
    var prev: ListNode?
    var next: ListNode?
    
    init(_ key: Int, _ value: Int) {
        self.key = key
        self.value = value
    }
}

class LRUCache {
    private let capacity: Int
    private var map: [Int: ListNode]
    private var head: ListNode  // Most recently used
    private var tail: ListNode  // Least recently used
    
    init(_ capacity: Int) {
        self.capacity = capacity
        self.map = [Int: ListNode]()
        self.head = ListNode(0, 0)
        self.tail = ListNode(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head
    }
    
    func get(_ key: Int) -> Int {
        guard let node = map[key] else { return -1 }
        moveToHead(node)
        return node.value
    }
    
    func put(_ key: Int, _ value: Int) {
        if let node = map[key] {
            node.value = value
            moveToHead(node)
        } else {
            let newNode = ListNode(key, value)
            map[key] = newNode
            addToHead(newNode)
            
            if map.count > capacity {
                let tailNode = removeTail()
                map.removeValue(forKey: tailNode.key)
            }
        }
    }
    
    private func moveToHead(_ node: ListNode) {
        removeNode(node)
        addToHead(node)
    }
    
    private func addToHead(_ node: ListNode) {
        node.prev = head
        node.next = head.next
        head.next?.prev = node
        head.next = node
    }
    
    private func removeNode(_ node: ListNode) {
        node.prev?.next = node.next
        node.next?.prev = node.prev
    }
    
    private func removeTail() -> ListNode {
        let tailNode = tail.prev!
        removeNode(tailNode)
        return tailNode
    }
}
```

## Recursion

- Always remember to always define a base case so that your recursion will end.
- Recursion is useful for permutation, because it generates all combinations and tree-based questions. You should know how to generate all permutations of a sequence as well as how to handle duplicates.
- Recursion implicitly uses a stack. Hence all recursive approaches can be rewritten iteratively using a stack. Beware of cases where the recursion level goes too deep and causes a stack overflow (the default limit in Python is 1000). You may get bonus points for pointing this out to the interviewer. Recursion will never be O(1) space complexity because a stack is involved, unless there is tail-call optimization (TCO). Find out if your chosen language supports TCO.
- Number of base cases - In the fibonacci recursion, note that one of the recursive calls invoke fib(n - 2). This indicates that you should have 2 base cases defined so that your code covers all possible invocations of the function within the input range. If your recursive function only invokes fn(n - 1), then only one base case is needed.

### Memoization
In some cases, you may be computing the result for previously computed inputs. Let's look at the Fibonacci example again. fib(5) calls fib(4) and fib(3), and fib(4) calls fib(3) and fib(2). fib(3) is being called twice! If the value for fib(3) is memoized and used again, that greatly improves the efficiency of the algorithm and the time complexity becomes O(n).

## Sorting

### Binary search
When a given sequence is in a sorted order (be it ascending or descending), using binary search should be one of the first things that come to your mind.

Template
```
function binarySearch(array, target):
    1. Define two pointers: "left" initialized to 0 and "right" initialized to (length of array - 1).
    
    2. While "left" is less than or equal to "right":
       a. Calculate the middle index: mid = (left + right) / 2.
       b. If array[mid] equals target:
          i. Return mid.
       c. If array[mid] is less than target:
          i. Set left = mid + 1.
       d. Else:
          i. Set right = mid - 1.

    3. Return "Not Found" or an equivalent indicator (like -1).
```

```swift
func binarySearch(_ arr: [Int], _ target: Int) -> Int {
    var left = 0
    var right = arr.count - 1

    while left <= right {
        let mid = (left + right) / 2

        if arr[mid] == target {
            return mid  // Target value found, return its index
        }

        if arr[mid] < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }

    return -1  // Target value not found in the array
}
```

If you want to find the closest value that's less than the target value in a sorted array, you can modify the binary search algorithm slightly. At the end of the standard binary search loop, the right pointer will indicate the position where the target should be if it were in the array (or before it).

You can use this property to get the closest value less than the target. After the loop ends, the value at `arr[right]` would be the closest value less than the target (if right is within the bounds of the array).

```swift
// White loop here
while (...) {
    // Your loop logic here
}

// Check if 'right' is within bounds
if right >= 0 {
    return arr[right]
}
```

### Sorting an input that has limited range
Counting sort is a non-comparison-based sort you can use on numbers where you know the range of values beforehand.

Template
```
function countingSort(inputArray, maxValue):
    1. Initialize an array "count" of zeros with a size of (maxValue + 1).
    2. For each element "x" in inputArray:
       a. Increment count[x] by 1.

    3. Initialize an output array "sortedArray" of the same size as inputArray.
    4. Initialize a position variable "pos" to 0.
    5. For each index "i" from 0 to maxValue:
       a. While count[i] is greater than 0:
          i. Place the value "i" in sortedArray[pos].
          ii. Increment pos by 1.
          iii. Decrement count[i] by 1.

    6. Return sortedArray.
```

Example
```swift
func countingSort(_ arr: [Int], _ maxValue: Int) -> [Int] {
    // Step 1: Initialize count array
    var count = Array(repeating: 0, count: maxValue + 1)

    // Step 2: Populate count array with frequencies
    for num in arr {
        count[num] += 1
    }

    // Step 3-5: Reconstruct the sorted array using the count array
    var sortedIndex = 0
    var sortedArray = Array(repeating: 0, count: arr.count)

    for i in 0..<count.count {
        while count[i] > 0 {
            sortedArray[sortedIndex] = i
            sortedIndex += 1
            count[i] -= 1
        }
    }

    // Step 6: Return the sorted array
    return sortedArray
}
```

## Matrix

### Create an empty X x M matrix:
```swift
let matrix = Array(repeating: Array(repeating: -1, count: M), count: X) // Replace -1 with your desired default value
```

### Transposing a matrix
The transpose of a matrix is found by interchanging its rows into columns or columns into rows.

Many grid-based games can be modeled as a matrix, such as Tic-Tac-Toe, Sudoku, Crossword, Connect 4, Battleship, etc. It is not uncommon to be asked to verify the winning condition of the game. For games like Tic-Tac-Toe, Connect 4 and Crosswords, where verification has to be done vertically and horizontally, one trick is to write code to verify the matrix for the horizontal cells, transpose the matrix, and reuse the logic for horizontal verification to verify originally vertical cells (which are now horizontal).

```swift
func transpose(_ matrix: [[Int]]) -> [[Int]] {
    // Create a new 2D array with dimensions columns x rows of the original matrix
    var transposed = Array(repeating: Array(repeating: 0, count: matrix.count), count: matrix[0].count)
    
    for i in 0..<matrix.count {
        for j in 0..<matrix[0].count {
            transposed[j][i] = matrix[i][j]
        }
    }
    
    return transposed
}
```

### Linked List

### Sentinel/dummy nodes
Adding a sentinel/dummy node at the head and/or tail might help to handle many edge cases where operations have to be performed at the head or the tail. The presence of dummy nodes essentially ensures that operations will never be done on the head or the tail, thereby removing a lot of headache in writing conditional checks to dealing with null pointers. Be sure to remember to remove them at the end of the operation.

Example
```swift
class ListNode {
    var val: Int
    var next: ListNode?

    init(_ val: Int = 0, _ next: ListNode? = nil) {
        self.val = val
        self.next = next
    }
}

func mergeTwoSortedLists(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {
    let dummy = ListNode(-1)  // Sentinel/dummy node
    var current = dummy  // Pointer to build the merged list
    var l1 = l1
    var l2 = l2

    while let left = l1, let right = l2 {
        if left.val < right.val {
            current.next = left
            l1 = left.next
        } else {
            current.next = right
            l2 = right.next
        }
        current = current.next!
    }

    // If there are remaining nodes in l1 or l2
    current.next = l1 ?? l2

    return dummy.next  // Return the next of dummy as the merged list's head
}
```

In the `mergeTwoSortedLists` function, we utilize a dummy node as the head of our merged list. By doing this, we don't have to write special logic to initialize the head of the merged list, because `dummy.next` will naturally point to the start of the merged list at the end of the process. The dummy node serves as a placeholder and helps in simplifying the code.

## Two pointers
Two pointer approaches are also common for linked lists. This approach is used for many classic linked list problems.

- Getting the kth from last node - Have two pointers, where one is k nodes ahead of the other. When the node ahead reaches the end, the other node is k nodes behind
- Detecting cycles - Have two pointers, where one pointer increments twice as much as the other, if the two pointers meet, means that there is a cycle
- Getting the middle node - Have two pointers, where one pointer increments twice as much as the other. When the faster node reaches the end of the list, the slower node will be at the middle

### Using space
Many linked list problems can be easily solved by creating a new linked list and adding nodes to the new linked list with the final result. However, this takes up extra space and makes the question much less challenging. The interviewer will usually request that you modify the linked list in-place and solve the problem without additional storage. You can borrow ideas from the Reverse a Linked List problem.

### Elegant modification operations
As mentioned earlier, a linked list's non-sequential nature of memory allows for efficient modification of its contents. Unlike arrays where you can only modify the value at a position, for linked lists you can also modify the next pointer in addition to the value.

Here are some common operations and how they can be achieved easily:
- Truncate a list - Set the next pointer to null at the last element
- Swapping values of nodes - Just like arrays, just swap the value of the two nodes, there's no need to swap the next pointer
- Combining two lists - attach the head of the second list to the tail of the first list

## Queue

Most languages don't have a built-in Queue class which can be used, and candidates often use arrays (JavaScript) or lists (Python) as a queue. However, note that the dequeue operation (assuming the front of the queue is on the left) in such a scenario will be O(n) because it requires shifting of all other elements left by one. In such cases, you can flag this to the interviewer and say that you assume that there's a queue data structure to use which has an efficient dequeue operation.

## Interval

### Sort the array of intervals by its starting point
A common routine for interval questions is to sort the array of intervals by each interval's starting value. This step is crucial to solving the Merge Intervals question.

### Checking if two intervals overlap
Be familiar with writing code to check if two intervals overlap.

```swift
func isOverlap(_ a: (Int, Int), _ b: (Int, Int)) -> Bool {
    return a.0 < b.1 && b.0 < a.1
}
```
Trick to remember: both the higher pos must be greater then both lower pos.

Merging two intervals
```swift
func mergeOverlappingIntervals(_ a: (Int, Int), _ b: (Int, Int)) -> (Int, Int) {
    return (min(a.0, b.0), max(a.1, b.1))
}
```

## Tree

![Tree](https://upload.wikimedia.org/wikipedia/commons/5/5e/Binary_tree_v2.svg)

### Traversals

- In-order traversal - Left -> Root -> Right.<br/>
Result: 2, 7, 5, 6, 11, 1, 9, 5, 9
- Pre-order traversal - Root -> Left -> Right<br/>
Result: 1, 7, 2, 6, 5, 11, 9, 9, 5
- Post-order traversal - Left -> Right -> Root<br/>
Result: 2, 5, 11, 6, 7, 5, 9, 9, 1

Example (Recursive)
```swift
class TreeNode {
    var val: Int
    var left: TreeNode?
    var right: TreeNode?

    init(_ val: Int = 0, _ left: TreeNode? = nil, _ right: TreeNode? = nil) {
        self.val = val
        self.left = left
        self.right = right
    }
}

// In-Order Traversal (Left, Root, Right)
func inOrderTraversal(_ root: TreeNode?) -> [Int] {
    var result: [Int] = []
    
    func helper(_ node: TreeNode?) {
        guard let node = node else { return }
        
        helper(node.left)
        result.append(node.val)
        helper(node.right)
    }

    helper(root)
    return result
}

// Pre-Order Traversal (Root, Left, Right)
func preOrderTraversal(_ root: TreeNode?) -> [Int] {
    var result: [Int] = []
    
    func helper(_ node: TreeNode?) {
        guard let node = node else { return }
        
        result.append(node.val)
        helper(node.left)
        helper(node.right)
    }

    helper(root)
    return result
}

// Post-Order Traversal (Left, Right, Root)
func postOrderTraversal(_ root: TreeNode?) -> [Int] {
    var result: [Int] = []
    
    func helper(_ node: TreeNode?) {
        guard let node = node else { return }
        
        helper(node.left)
        helper(node.right)
        result.append(node.val)
    }

    helper(root)
    return result
}
```

Example (Iterative)
```swift
class TreeNode {
    var val: Int
    var left: TreeNode?
    var right: TreeNode?

    init(_ val: Int = 0, _ left: TreeNode? = nil, _ right: TreeNode? = nil) {
        self.val = val
        self.left = left
        self.right = right
    }
}

// In-Order Traversal (Left, Root, Right)
func inOrderTraversal(_ root: TreeNode?) -> [Int] {
    var result: [Int] = []
    var stack: [TreeNode] = []
    var current = root

    while current != nil || !stack.isEmpty {
        while let node = current {
            stack.append(node)
            current = node.left
        }

        current = stack.removeLast()
        result.append(current!.val)
        current = current!.right
    }

    return result
}

// Pre-Order Traversal (Root, Left, Right)
func preOrderTraversal(_ root: TreeNode?) -> [Int] {
    var result: [Int] = []
    var stack: [TreeNode] = []
    if let root = root { stack.append(root) }

    while !stack.isEmpty {
        let current = stack.removeLast()
        result.append(current.val)

        if let right = current.right { stack.append(right) }
        if let left = current.left { stack.append(left) }
    }

    return result
}

// Post-Order Traversal (Left, Right, Root)
func postOrderTraversal(_ root: TreeNode?) -> [Int] {
    var result: [Int] = []
    var stack: [TreeNode] = []
    var lastVisited: TreeNode? = nil
    var current = root

    while current != nil || !stack.isEmpty {
        while let node = current {
            stack.append(node)
            current = node.left
        }

        if let peekNode = stack.last {
            if peekNode.right == nil || peekNode.right === lastVisited {
                result.append(peekNode.val)
                lastVisited = stack.removeLast()
            } else {
                current = peekNode.right
            }
        }
    }

    return result
}
```

### Binary search tree (BST)
In-order traversal of a BST will give you all elements in order.

Be very familiar with the properties of a BST and validating that a binary tree is a BST. This comes up more often than expected.

When a question involves a BST, the interviewer is usually looking for a solution which runs faster than O(n).

**Time complexity**<br/>
Operation	Big-O<br/>
Access	O(log(n))<br/>
Search	O(log(n))<br/>
Insert	O(log(n))<br/>
Remove	O(log(n))

Space complexity of traversing balanced trees is O(h) where h is the height of the tree, while traversing very skewed trees (which is essentially a linked list) will be O(n).

### Use recursion
Recursion is the most common approach for traversing trees. When you notice that the subtree problem can be used to solve the entire problem, try using recursion.

When using recursion, always remember to check for the base case, usually where the node is `null`.

Sometimes it is possible that your recursive function needs to return two values.

### Traversing by level
When you are asked to traverse a tree by level, use breadth-first search.

## Graph

### Graph representations
You can be given a list of edges and you have to build your own graph from the edges so that you can perform a traversal on them. The common graph representations are:

- Adjacency matrix
- Adjacency list
- Hash table of hash tables

Using a hash table of hash tables would be the simplest approach during algorithm interviews. It will be rare that you have to use an adjacency matrix or list for graph questions during interviews.

```swift
typealias Node = String
typealias Graph = [Node: [Node: Bool]]

func addEdge(_ graph: inout Graph, from: Node, to: Node) {
    if graph[from] == nil {
        graph[from] = [:]
    }
    graph[from]![to] = true
}

func buildGraph(_ edges: [(Node, Node)]) -> Graph {
    var graph = Graph()

    for (from, to) in edges {
        addEdge(&graph, from: from, to: to)
        addEdge(&graph, from: to, to: from) // For undirected graph. Remove this for directed graph.
    }

    return graph
}

// Test
let edges: [(Node, Node)] = [
    ("A", "B"),
    ("A", "C"),
    ("B", "D"),
    ("C", "D"),
    ("D", "E")
]

let graph = buildGraph(edges)
print(graph)
```

In algorithm interviews, graphs are commonly given in the input as 2D matrices where cells are the nodes and each cell can traverse to its adjacent cells (up/down/left/right). Hence it is important that you be familiar with traversing a 2D matrix. When traversing the matrix, always ensure that your current position is within the boundary of the matrix and has not been visited before.

```swift
typealias Matrix = [[Int]]

func traverseMatrix(_ matrix: Matrix) {
    guard !matrix.isEmpty, !matrix[0].isEmpty else {
        return
    }

    let rows = matrix.count
    let cols = matrix[0].count
    var visited = Array(repeating: Array(repeating: false, count: cols), count: rows)

    func isValid(row: Int, col: Int) -> Bool {
        return row >= 0 && row < rows && col >= 0 && col < cols && !visited[row][col]
    }

    func dfs(row: Int, col: Int) {
        guard isValid(row: row, col: col) else { return }

        print(matrix[row][col]) // Process the current cell
        visited[row][col] = true

        // Traverse up/down/left/right
        dfs(row: row - 1, col: col) // Up
        dfs(row: row + 1, col: col) // Down
        dfs(row: row, col: col - 1) // Left
        dfs(row: row, col: col + 1) // Right
    }

    for i in 0..<rows {
        for j in 0..<cols {
            if !visited[i][j] {
                dfs(row: i, col: j)
            }
        }
    }
}

// Test
let matrix: Matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

traverseMatrix(matrix)
```

### 2D Matrix as a graph

Transforming the problem of traversing a 2D matrix into a graph problem involves viewing each cell of the matrix as a node of a graph and the possible movements (e.g., up, down, left, right) from a cell as edges connecting the nodes.

Here's a step-by-step guide on how you can transform the problem:

1. **Nodes Representation**: Each cell in the matrix, identified by its coordinates (i, j), can be treated as a node in the graph.

2. **Edges Representation**: For each cell, consider its adjacent cells. An edge exists between two nodes if you can move from one cell to the adjacent cell. For instance, if movements are restricted to up, down, left, and right, then:
   - The node (i, j) will have an edge to (i-1, j) if (i-1, j) is a valid cell (i.e., moving upwards).
   - The node (i, j) will have an edge to (i+1, j) if (i+1, j) is a valid cell (i.e., moving downwards).
   - The node (i, j) will have an edge to (i, j-1) if (i, j-1) is a valid cell (i.e., moving left).
   - The node (i, j) will have an edge to (i, j+1) if (i, j+1) is a valid cell (i.e., moving right).

3. **Graph Representation**: There are several ways to represent a graph, such as an adjacency list, adjacency matrix, or a hash table of hash tables. In the context of a matrix traversal, an adjacency list is often a good fit. Each node (i.e., matrix cell) would have a list of its adjacent nodes.

4. **Traversal**: With the graph constructed, you can apply standard graph traversal algorithms like DFS or BFS to explore the nodes. 

5. **Special Conditions**: If there are certain cells in the matrix that you cannot traverse (like obstacles), you simply skip adding them as valid edges in the graph representation. 

This transformation is particularly useful when you have more complex conditions for traversal, or when the problem involves finding paths, connected components, etc. It allows you to leverage a wide range of graph algorithms to solve matrix-based problems.

Example
```swift
typealias Node = (Int, Int)
typealias Graph = [String: [Node]]

func matrixToGraph(_ matrix: [[Int]]) -> Graph {
    let rows = matrix.count
    let cols = matrix[0].count
    var graph: Graph = [:]

    func getNodeKey(_ node: Node) -> String {
        return "\(node.0),\(node.1)"
    }

    func getAdjacentNodes(_ row: Int, _ col: Int) -> [Node] {
        let directions: [Node] = [(-1, 0), (1, 0), (0, -1), (0, 1)]  // Up, Down, Left, Right
        var adjacentNodes: [Node] = []

        for (dx, dy) in directions {
            let newRow = row + dx
            let newCol = col + dy

            if newRow >= 0 && newRow < rows && newCol >= 0 && newCol < cols {
                adjacentNodes.append((newRow, newCol))
            }
        }

        return adjacentNodes
    }

    for i in 0..<rows {
        for j in 0..<cols {
            let node: Node = (i, j)
            let key = getNodeKey(node)
            graph[key] = getAdjacentNodes(i, j)
        }
    }

    return graph
}

func traverseGraph(_ graph: Graph, startNode: Node) {
    var visited: Set<String> = []

    func dfs(_ node: Node) {
        let key = "\(node.0),\(node.1)"
        if visited.contains(key) { return }

        print(node)
        visited.insert(key)

        let neighbors = graph[key] ?? []
        for neighbor in neighbors {
            dfs(neighbor)
        }
    }

    dfs(startNode)
}

// Test
let matrix: [[Int]] = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

let graph = matrixToGraph(matrix)
traverseGraph(graph, startNode: (0, 0))  // Start traversal from top-left corner
```

### Time complexity
|V| is the number of vertices while |E| is the number of edges.

Algorithm	Big-O<br/>
Depth-first search	O(|V| + |E|)<br/>
Breadth-first search	O(|V| + |E|)<br/>
Topological sort	O(|V| + |E|)<br/>

### Notes
A tree-like diagram could very well be a graph that allows for cycles and a naive recursive solution would not work. In that case you will have to handle cycles and keep a set of visited nodes when traversing.

Ensure you are correctly keeping track of visited nodes and not visiting each node more than once. Otherwise your code could end up in an infinite loop.

### Depth-first search

```swift
func dfs(_ matrix: [[Int]]) {
    // Check for an empty matrix/graph
    guard !matrix.isEmpty, !matrix[0].isEmpty else { return }

    let rows = matrix.count
    let cols = matrix[0].count
    var visited = Set<(Int, Int)>()
    let directions: [(Int, Int)] = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    func traverse(_ i: Int, _ j: Int) {
        if visited.contains((i, j)) {
            return
        }

        visited.insert((i, j))

        // Traverse neighbors
        for direction in directions {
            let nextI = i + direction.0
            let nextJ = j + direction.1

            if nextI >= 0, nextI < rows, nextJ >= 0, nextJ < cols {
                // Add any question-specific checks here if needed
                traverse(nextI, nextJ)
            }
        }
    }

    for i in 0..<rows {
        for j in 0..<cols {
            traverse(i, j)
        }
    }
}
```

### Breadth-first search

```swift
import Foundation

func bfs(_ matrix: [[Int]]) {
    // Check for an empty matrix/graph
    guard !matrix.isEmpty, !matrix[0].isEmpty else { return }

    let rows = matrix.count
    let cols = matrix[0].count
    var visited = Set<(Int, Int)>()
    let directions: [(Int, Int)] = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    func traverse(_ i: Int, _ j: Int) {
        var queue: [(Int, Int)] = [(i, j)]
        
        while !queue.isEmpty {
            let (currI, currJ) = queue.removeFirst()

            if !visited.contains((currI, currJ)) {
                visited.insert((currI, currJ))

                // Traverse neighbors
                for direction in directions {
                    let nextI = currI + direction.0
                    let nextJ = currJ + direction.1

                    if nextI >= 0, nextI < rows, nextJ >= 0, nextJ < cols {
                        // Add any question-specific checks here if needed
                        queue.append((nextI, nextJ))
                    }
                }
            }
        }
    }

    for i in 0..<rows {
        for j in 0..<cols {
            traverse(i, j)
        }
    }
}
```

### Topological sorting

A topological sort or topological ordering of a directed graph is a linear ordering of its vertices such that for every directed edge uv from vertex u to vertex v, u comes before v in the ordering. Precisely, a topological sort is a graph traversal in which each node v is visited only after all its dependencies are visited.

Topological sorting is most commonly used for scheduling a sequence of jobs or tasks which has dependencies on other jobs/tasks. The jobs are represented by vertices, and there is an edge from x to y if job x must be completed before job y can be started.

Another example is taking courses in university where courses have pre-requisites.

In the context of directed graphs, in-degree of a node refers to the number of incoming edges to that node. In simpler terms, it's a count of how many vertices "point to" or "lead to" the given vertex.

```swift
typealias Graph = [Int: [Int]]

func topologicalSort(_ graph: Graph) -> [Int]? {
    let numNodes = graph.count
    var inDegree = [Int: Int]()
    var zeroInDegreeQueue: [Int] = []
    var result: [Int] = []

    // Initialize in-degree counts
    for node in graph.keys {
        inDegree[node] = 0
    }

    for neighbors in graph.values {
        for neighbor in neighbors {
            inDegree[neighbor, default: 0] += 1
        }
    }

    // Find all nodes with zero in-degree
    for (node, degree) in inDegree {
        if degree == 0 {
            zeroInDegreeQueue.append(node)
        }
    }

    while !zeroInDegreeQueue.isEmpty {
        let current = zeroInDegreeQueue.removeFirst()
        result.append(current)

        let neighbors = graph[current] ?? []
        for neighbor in neighbors {
            inDegree[neighbor]! -= 1
            if inDegree[neighbor] == 0 {
                zeroInDegreeQueue.append(neighbor)
            }
        }
    }

    // If result does not include all nodes, there's a cycle in the graph
    return result.count == numNodes ? result : nil
}

// Test
let graph: Graph = [
    5: [2],
    4: [0, 2],
    2: [3],
    3: [1],
    1: [],
    0: []
]

if let sortedOrder = topologicalSort(graph) {
    print(sortedOrder)  // Possible output: [5, 4, 2, 3, 1, 0]
} else {
    print("Graph has a cycle")
}
```

## Heap

A heap is a specialized tree-based data structure which is a complete tree that satisfies the heap property.

- Max heap - In a max heap, the value of a node must be greatest among the node values in its entire subtree. The same property must be recursively true for all nodes in the tree.
- Min heap - In a min heap, the value of a node must be smallest among the node values in its entire subtree. The same property must be recursively true for all nodes in the tree.

In the context of algorithm interviews, heaps and priority queues can be treated as the same data structure. A heap is a useful data structure when it is necessary to repeatedly remove the object with the highest (or lowest) priority, or when insertions need to be interspersed with removals of the root node.

### Time complexity
Operation	Big-O<br/>
Find max/min	O(1)<br/>
Insert	O(log(n))<br/>
Remove	O(log(n))<br/>
Heapify (create a heap out of given array of elements)	O(n)<br/>

### Mention of k
If you see a top or lowest k being mentioned in the question, it is usually a signal that a heap can be used to solve the problem, such as in Top K Frequent Elements.

If you require the top k elements use a Min Heap of size k. Iterate through each element, pushing it into the heap (for python heapq, invert the value before pushing to find the max). Whenever the heap size exceeds k, remove the minimum element, that will guarantee that you have the k largest elements.

## Trie

Tries are special trees (prefix trees) that make searching and storing strings more efficient. Tries have many practical applications, such as conducting searches and providing autocomplete. It is helpful to know these common applications so that you can easily identify when a problem can be efficiently solved using a trie.

Be familiar with implementing from scratch, a Trie class and its add, remove and search methods.

Example:
```swift
class TrieNode {
    var children: [Character: TrieNode] = [:]
    var isEndOfWord: Bool = false
}

class Trie {
    private let root = TrieNode()

    // Inserts a word into the trie
    func insert(_ word: String) {
        var currentNode = root
        for char in word {
            if currentNode.children[char] == nil {
                currentNode.children[char] = TrieNode()
            }
            currentNode = currentNode.children[char]!
        }
        currentNode.isEndOfWord = true
    }

    // Returns if the word is in the trie
    func search(_ word: String) -> Bool {
        var currentNode = root
        for char in word {
            guard let nextNode = currentNode.children[char] else {
                return false
            }
            currentNode = nextNode
        }
        return currentNode.isEndOfWord
    }

    // Returns if there's any word in the trie that starts with the given prefix
    func startsWith(_ prefix: String) -> Bool {
        var currentNode = root
        for char in prefix {
            guard let nextNode = currentNode.children[char] else {
                return false
            }
            currentNode = nextNode
        }
        return true
    }
}

// Test
let trie = Trie()
trie.insert("apple")
print(trie.search("apple"))    // Expected output: true
print(trie.search("app"))      // Expected output: false
print(trie.startsWith("app"))  // Expected output: true
trie.insert("app")
print(trie.search("app"))      // Expected output: true
```

### Time complexity
`m` is the length of the string used in the operation.

Operation	Big-O<br/>
Search	O(m)<br/>
Insert	O(m)<br/>
Remove	O(m)<br/>

### Techniques
Sometimes preprocessing a dictionary of words (given in a list) into a trie, will improve the efficiency of searching for a word of length k, among n words. Searching becomes O(k) instead of O(n).

## Geometry

Distance between two points
```swift
struct Point {
    var x: Double
    var y: Double
}

func distanceBetweenTwoPoints(_ point1: Point, _ point2: Point) -> Double {
    return sqrt(pow(point2.x - point1.x, 2) + pow(point2.y - point1.y, 2))
}
```

Overlapping Circles
```swift
struct Circle {
    var center: Point
    var radius: Double
}

func areCirclesOverlapping(_ circle1: Circle, _ circle2: Circle) -> Bool {
    let distance = distanceBetweenTwoPoints(circle1.center, circle2.center)
    return distance <= (circle1.radius + circle2.radius)
}
```

Overlapping Rectanges
```swift
struct Rectangle {
    var topLeft: Point
    var bottomRight: Point
}

func areRectanglesOverlapping(_ rect1: Rectangle, _ rect2: Rectangle) -> Bool {
    // Check if one rectangle is to the left of the other
    if rect1.topLeft.x > rect2.bottomRight.x || rect2.topLeft.x > rect1.bottomRight.x {
        return false
    }

    // Check if one rectangle is above the other
    if rect1.topLeft.y < rect2.bottomRight.y || rect2.topLeft.y < rect1.bottomRight.y {
        return false
    }

    return true // If none of the above cases occurred, rectangles are overlapping
}
```

# Notorious Problems

## Robot Room Cleaner
Link: https://leetcode.com/problems/robot-room-cleaner/description/

```swift
class Robot {
    func move() -> Bool { return true }  // Placeholder for the actual implementation
    func turnRight() {}                  // Placeholder for the actual implementation
    func turnLeft() {}                   // Placeholder for the actual implementation
    func clean() {}                      // Placeholder for the actual implementation
}

func cleanRoom(_ robot: Robot) {
    let dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    var visited = Set<String>()
    
    func goBack() {
        robot.turnRight()
        robot.turnRight()
        _ = robot.move()
        robot.turnRight()
        robot.turnRight()
    }
    
    func backtrack(_ row: Int, _ col: Int, _ d: Int) {
        visited.insert("\(row),\(col)")
        robot.clean()
        
        for i in 0..<4 {
            let newD = (d + i) % 4
            let newRow = row + dirs[newD].0
            let newCol = col + dirs[newD].1
            
            if !visited.contains("\(newRow),\(newCol)") && robot.move() {
                backtrack(newRow, newCol, newD)
                goBack()
            }
            
            robot.turnRight()
        }
    }
    
    backtrack(0, 0, 0)
}
```

## Kadane's Algorithm

Kadane's algorithm is used to find the maximum sum of a contiguous subarray within a one-dimensional array of numbers.

```swift
func maxSubArraySum(_ arr: [Int]) -> Int {
    guard !arr.isEmpty else { return 0 }

    var maxCurrent = arr[0]
    var maxGlobal = arr[0]

    for i in 1..<arr.count {
        maxCurrent = max(arr[i], maxCurrent + arr[i])
        maxGlobal = max(maxGlobal, maxCurrent)
    }

    return maxGlobal
}

// Example usage:
let numbers = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
print(maxSubArraySum(numbers))  // Outputs: 6 (because [4, -1, 2, 1] has the largest sum)
```

If you want to track the start and end indices of the subarray with the largest sum, you can use the following code:

```swift
func maxSubArraySumWithIndices(_ arr: [Int]) -> (sum: Int, start: Int, end: Int) {
    guard !arr.isEmpty else { return (sum: 0, start: -1, end: -1) }

    var maxCurrent = arr[0]
    var maxGlobal = arr[0]
    var start = 0
    var end = 0
    var tempStart = 0

    for i in 1..<arr.count {
        if arr[i] > maxCurrent + arr[i] {
            maxCurrent = arr[i]
            tempStart = i
        } else {
            maxCurrent += arr[i]
        }

        if maxCurrent > maxGlobal {
            maxGlobal = maxCurrent
            start = tempStart
            end = i
        }
    }

    return (sum: maxGlobal, start: start, end: end)
}

// Example usage:
let numbers = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
let result = maxSubArraySumWithIndices(numbers)
print("Max sum is: \(result.sum) from index \(result.start) to \(result.end)")
// Outputs: Max sum is: 6 from index 3 to 6 (because [4, -1, 2, 1] has the largest sum)
```

## Great Explanation of Dynamic Programming (DP)

- https://leetcode.com/problems/word-break-ii/editorial/

## Disjoint Union Set (DSU)

```swift
class DisjointSetUnion {
    private var parent: [Int]

    init(_ size: Int) {
        parent = Array(0..<size)
    }

    // Find the root of a node
    func find(_ i: Int) -> Int {
        if parent[i] != i {
            parent[i] = find(parent[i])
        }
        return parent[i]
    }

    // Union the two nodes
    func union(_ i: Int, _ j: Int) {
        parent[find(i)] = find(j)
    }
}

func smallestStringWithSwaps(_ s: String, _ pairs: [[Int]]) -> String {
    let n = s.count
    let dsu = DisjointSetUnion(n)
    let chars = Array(s)

    // Create the disjoint set using the pairs
    for pair in pairs {
        dsu.union(pair[0], pair[1])
    }

    // Create mapping of root -> [indices]
    var indexGroups: [Int: [Int]] = [:]
    for i in 0..<n {
        let root = dsu.find(i)
        indexGroups[root, default: []].append(i)
    }

    // Sort characters within each group and rebuild the result
    var result = Array(chars)
    for indices in indexGroups.values {
        let sortedChars = indices.map { chars[$0] }.sorted()
        for (i, index) in indices.enumerated() {
            result[index] = sortedChars[i]
        }
    }

    return String(result)
}

// Test cases
print(smallestStringWithSwaps("dcab", [[0, 3], [1, 2]])) // "bacd"
print(smallestStringWithSwaps("cba", [[0, 1], [1, 2]]))  // "abc"
print(smallestStringWithSwaps("cbfdae", [[0, 1], [3, 2], [5, 2], [1, 4]])) // "abdecf"
```

Let's dry run the `smallestStringWithSwaps` function with the input string `s = "dcab"` and pairs `[[0,3],[1,2]]`.

**Initialization**:
```swift
var parent = Array(0..<s.count)
```
This initializes each character's index as its own parent:
`parent = [0, 1, 2, 3]`

**Processing the pairs using union**:

1. For the pair `[0,3]`:
   - The union function merges the sets containing `0` and `3`.
   - `parent` becomes: `[3, 1, 2, 3]`

2. For the pair `[1,2]`:
   - The union function merges the sets containing `1` and `2`.
   - `parent` becomes: `[3, 2, 2, 3]`

**Create mapping of root to its indices**:
We want to know which indices belong to which set. 
```swift
var indexGroups: [Int: [Int]] = [:]
for i in 0..<s.count {
    let root = find(i)
    indexGroups[root, default: []].append(i)
}
```
After this loop, `indexGroups` becomes:
```
{
    3: [0, 3],
    2: [1, 2]
}
```
This means characters at indices `0` and `3` can be swapped with each other, and characters at indices `1` and `2` can be swapped with each other.

**Reorder characters within each set**:
We want the characters in each group to be sorted in ascending order to make the string lexicographically smallest.
```swift
var chars = Array(s)
for indices in indexGroups.values {
    let sortedChars = indices.map { chars[$0] }.sorted()
    for (i, index) in indices.enumerated() {
        chars[index] = sortedChars[i]
    }
}
```
1. For `indices = [0, 3]` (i.e., characters `'d'` and `'b'`):
   - Sort them to get `'b'` and `'d'`.
   - Update the characters in `chars` to reflect the new order: `chars` becomes `['b', 'c', 'a', 'd']`.

2. For `indices = [1, 2]` (i.e., characters `'c'` and `'a'`):
   - Sort them to get `'a'` and `'c'`.
   - Update the characters in `chars` to reflect the new order: `chars` becomes `['b', 'a', 'c', 'd']`.

**Return the modified string**:
The final step is to join the `chars` array:
```swift
return String(chars)
```
This returns the string `"bacd"`.

So, for the input `s = "dcab"` and pairs `[[0,3],[1,2]]`, the function `smallestStringWithSwaps` returns the string `"bacd"`.

## Monotonic Stack

A monotonic stack is a type of stack that maintains a certain monotonicity property as elements are pushed into it. Depending on the problem at hand, this might be a strictly increasing or strictly decreasing stack. In other words, when you push a new element onto the stack, the stack ensures that its elements are still in either non-decreasing or non-increasing order by potentially popping off the elements that violate the monotonicity.

The primary purpose of a monotonic stack is to efficiently answer questions like:

- For each element in an array, what's the nearest smaller (or larger) element on the left or right of it?
- Finding the maximum rectangle in a histogram.

Let's dive into an example:

**Problem Statement:**
Given an array of numbers, for each element, find the first larger number to its right.

**Example:**
Input: `[4, 3, 2, 5, 1]`
Output: `[5, 5, 5, -1, -1]`

**Explanation:** 
For `4`, the next larger number is `5`.
For `3`, the next larger number is `5`.
For `2`, the next larger number is `5`.
For `5`, there is no larger number, hence `-1`.
For `1`, there is no larger number, hence `-1`.

**Swift Solution using Monotonic Stack:**
```swift
func nextLargerElement(_ nums: [Int]) -> [Int] {
    var result = Array(repeating: -1, count: nums.count) // initialize result array with -1s
    var stack: [Int] = [] // this will store indices

    for i in 0..<nums.count {
        while !stack.isEmpty && nums[i] > nums[stack.last!] {
            let idx = stack.removeLast() // get the last index from the stack
            result[idx] = nums[i] // found the next greater element for the element at index `idx`
        }
        stack.append(i) // push the current index to the stack
    }

    return result
}

// Testing the function
let input = [4, 3, 2, 5, 1]
print(nextLargerElement(input)) // Expected output: [5, 5, 5, -1, -1]
```

In this solution, we're keeping the stack in non-decreasing order. When a larger number is found, we keep popping from the stack until we find a number that's greater than the current one or the stack becomes empty. This helps in finding the next larger number for all the numbers that are smaller than the current number.

## Peak Valley (Arrays)

You are given an integer array prices where prices[i] is the price of a given stock on the ith day.

On each day, you may decide to buy and/or sell the stock. You can only hold at most one share of the stock at any time. However, you can buy it then immediately sell it on the same day.

Find and return the maximum profit you can achieve.

```
Input: prices = [7,1,5,3,6,4]
Output: 7
Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.
Total profit is 4 + 3 = 7.
```

```swift
class Solution {
    func maxProfit(_ prices: [Int]) -> Int {
        var i = 0
        var maxProfit = 0
        var valley = prices[0]
        var peak = prices[0]

        while i < prices.count - 1 {
            // Find the next valley
            while i < prices.count - 1 && prices[i] >= prices[i + 1] {
                i += 1
            }
            valley = prices[i]

            // Find the next peak
            while i < prices.count - 1 && prices[i] <= prices[i + 1] {
                i += 1
            }
            peak = prices[i]

            // Accumulate profit
            maxProfit += peak - valley
        }
        
        return maxProfit
    }
}

// Example usage:
let solution = Solution()
let prices = [7, 1, 5, 3, 6, 4]
print(solution.maxProfit(prices)) // Expected output: 7 (buy at 1, sell at 6)
```

## Dijkstra's algorithm

### Explanation:
Dijkstra's algorithm is a graph search algorithm that solves the single-source shortest path problem for a graph with non-negative edge path costs, producing the shortest path tree. This algorithm is often used in routing and as a subroutine in other graph algorithms.

Here are the basic steps:

1. **Initialization**: 
   - Set a tentative distance value for every node: set the initial node's distance to zero and all other nodes' distance to infinity. 
   - Set the initial node as the current node.
   - Mark all nodes as unvisited. Create a set of all the unvisited nodes.

2. **Main loop**:
   - For the current node, consider all of its neighbors and calculate their tentative distances through the current node. Compare the newly calculated tentative distance to the current assigned value and update the distance if the new value is smaller.
   - Once you have considered all of the neighbors of the current node, mark the current node as visited. A visited node will not be checked again.
   - Select the unvisited node with the smallest tentative distance, and set it as the new current node. Then go back to the previous step.

3. **Completion**:
   - The algorithm ends when every node has been visited. The algorithm has now constructed the shortest path tree from the source node to all other nodes.

### Swift Implementation:
```swift
typealias Graph = [String: [String: Int]]

func dijkstra(graph: Graph, start: String) -> (distances: [String: Int], previous: [String: String?]) {
    var distances: [String: Int] = [:]
    var previous: [String: String?] = [:]
    var unvisitedNodes = Array(graph.keys)
    
    // Initialize distances and previous nodes
    for node in unvisitedNodes {
        distances[node] = Int.max
        previous[node] = nil
    }
    distances[start] = 0

    // Dijkstra's algorithm
    while !unvisitedNodes.isEmpty {
        // Sort unvisited nodes by distance and remove the node with the smallest distance
        unvisitedNodes.sort { distances[$0, default: Int.max] < distances[$1, default: Int.max] }
        let currentNode = unvisitedNodes.removeFirst()

        // Update distances for neighbors
        if let neighbors = graph[currentNode] {
            for (neighbor, weight) in neighbors {
                let newDistance = distances[currentNode, default: Int.max] + weight
                if newDistance < distances[neighbor, default: Int.max] {
                    distances[neighbor] = newDistance
                    previous[neighbor] = currentNode
                }
            }
        }
    }

    return (distances, previous)
}

// Example Usage
let exampleGraph: Graph = [
    "A": ["B": 1, "D": 3],
    "B": ["A": 1, "D": 2, "E": 5],
    "D": ["A": 3, "B": 2, "E": 1],
    "E": ["B": 5, "D": 1]
]

let result = dijkstra(graph: exampleGraph, start: "A")
print("Distances:", result.distances)
print("Previous:", result.previous)
```

### Dry Run:
Let's dry run the algorithm using the `exampleGraph`:

1. **Initialization**: 
   - `distances = { A: 0, B: Infinity, D: Infinity, E: Infinity }`
   - `previous = { A: null, B: null, D: null, E: null }`
   - Current node = `A`
   - Unvisited nodes = `A, B, D, E`

2. **First Iteration**:
   - Current Node: `A`
   - Neighbors: `B` and `D`
     - For `B`: New distance = `0 + 1 = 1`, which is less than `Infinity`
       - `distances[B] = 1`, `previous[B] = A`
     - For `D`: New distance = `0 + 3 = 3`, which is less than `Infinity`
       - `distances[D] = 3`, `previous[D] = A`
   - Mark `A` as visited.
   - New current node = `B` (smallest distance among unvisited nodes)

3. **Second Iteration**:
   - Current Node: `B`
   - Neighbors: `A`, `D`, and `E`
     - For `D`: New distance = `1 + 2 = 3`, which is not less than current `3`
     - For `E`: New distance = `1 + 5 = 6`, which is less than `Infinity`
       - `distances[E] = 6`, `previous[E] = B`
   - Mark `B` as visited.
   - New current node = `D` (smallest distance among unvisited nodes)

4. **Third Iteration**:
   - Current Node: `D`
   - Neighbors: `A`, `B`, and `E`
     - For `E`: New distance = `3 + 1 = 4`, which is less than current `6`
       - `distances[E] = 4`, `previous[E] = D`
   - Mark `D` as visited.
   - New current node = `E` (last unvisited node)

5. **Fourth Iteration**:
   - Current Node: `E`
   - No updates since all neighbors have been visited.
   - Mark `E` as visited.

**Completion**:
`distances` = { A: 0, B: 1, D: 3, E: 4 }
`previous` = { A: null, B: 'A', D: 'A', E: 'D' }

The shortest path from A to E is `A -> B -> D -> E` with a distance of `4`.


### Time and Space Complexity:

Dijkstra's algorithm's time and space complexity are primarily influenced by the data structures used to implement it. The pseudocode provided earlier uses a basic array for the unvisited set and iterates over it to find the node with the smallest distance, which isn't the most efficient approach. Let's break down the complexities for the provided Swift implementation and then discuss how it can be optimized.

### For the provided Swift implementation:

**Time Complexity:**
1. Initializing `distances`, `previous`, and `unvisitedNodes` is \(O(V)\), where \(V\) is the number of vertices.
2. In the worst-case scenario, the `while` loop runs for every node, i.e., \(O(V)\).
3. Inside the `while` loop, the `sort` operation is \(O(V \log V)\).
4. Additionally, inside the `while` loop, we might go through all the edges of a node in the nested `for` loop, i.e., \(O(E)\) where \(E\) is the number of edges.

Multiplying these together, the worst-case time complexity is:
\[O(V \times (V \log V + E))\]
This can be approximated as \(O(V^2 \log V)\) in dense graphs where \(E \approx V^2\) and \(O(V^2)\) in sparse graphs.

**Space Complexity:**
1. The `distances` and `previous` maps have space complexity \(O(V)\).
2. The `unvisitedNodes` array also has space complexity \(O(V)\).

Summing these up, the overall space complexity is:
\[O(V)\]

### Optimized Version using Priority Queue:

You can optimize the time complexity by using a priority queue (or a binary heap) to manage the unvisited nodes. This would allow you to efficiently extract the node with the smallest tentative distance without having to sort the entire set of unvisited nodes each time.

With this optimization:

**Time Complexity:**
1. Initialization is still \(O(V)\).
2. The loop runs for every node and edge, i.e., \(O(V + E)\).
3. Extracting the minimum node from a priority queue is \(O(\log V)\).

Multiplying these together, the worst-case time complexity becomes:
\[O((V + E) \log V)\]

For a dense graph, this still reduces to \(O(V^2 \log V)\), but the constant factors are typically much better than the array-based approach. For sparse graphs, this is much faster, at \(O(V \log V)\).

**Space Complexity:**
The space complexity remains largely unchanged at \(O(V)\) for the data structures in the algorithm. However, if you include the priority queue's internal structures, it can go up slightly but remains within \(O(V)\) bounds.

In summary, while the naive version can be quite slow for large graphs, using a priority queue can significantly speed up Dijkstra's algorithm.

## Sorting

### Quick Sort

Quick Sort is a divide-and-conquer algorithm that works on the principle of choosing a 'pivot' element from the array and partitioning the other elements into two sub-arrays, according to whether they are less than or greater than the pivot. The sub-arrays are then sorted recursively.

1. **Choose a Pivot**: Select an element from the array as the pivot. This can be done in various ways, such as choosing the first element, the last element, the middle element, or even a random element.

2. **Partitioning**: Reorder the array so that all elements with values less than the pivot come before the pivot, while all elements with values greater than the pivot come after it. After this partitioning, the pivot is in its final position.

3. **Recursive Sorting**: Recursively apply the above steps to the sub-array of elements with smaller values and the sub-array of elements with greater values.

4. **Base Case**: The recursion base case is an array with zero or one element, which doesn't need to be sorted.

Now, here's an example of Quick Sort implemented in Swift:

```swift
func quickSort(_ arr: inout [Int], low: Int, high: Int) {
    if low < high {
        // Partition the array and get the pivot index
        let pi = partition(&arr, low: low, high: high)

        // Recursively sort elements before and after partition
        quickSort(&arr, low: low, high: pi - 1)
        quickSort(&arr, low: pi + 1, high: high)
    }
}

func partition(_ arr: inout [Int], low: Int, high: Int) -> Int {
    // Choose the rightmost element as pivot
    let pivot = arr[high]

    // Pointer for greater element
    var i = low - 1

    // Traverse through all elements and compare each with the pivot
    for j in low..<high {
        if arr[j] < pivot {
            // If element smaller than pivot is found, swap it with the greater element pointed by i
            i += 1
            arr.swapAt(i, j)
        }
    }

    // Swap the pivot element with the element at i + 1
    arr.swapAt(i + 1, high)

    // Return the position from where partition is done
    return i + 1
}

// Helper function to initiate QuickSort
func quickSortHelper(_ arr: [Int]) -> [Int] {
    var arr = arr
    quickSort(&arr, low: 0, high: arr.count - 1)
    return arr
}

// Example usage
let arr = [10, 7, 8, 9, 1, 5]
print("Original Array:", arr)
let sortedArray = quickSortHelper(arr)
print("Sorted Array:", sortedArray)
```

