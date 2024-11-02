
# Swift Algorithm Cheat Sheet

## Sliding Window - Maximum Sum Subarray of Size k

```swift
func maxSumSubarray(arr: [Int], k: Int) -> Int {
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

## Two Pointers - Pair of Elements Sum to Target in Sorted Array

```swift
func twoSumSorted(arr: [Int], target: Int) -> (Int, Int)? {
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

## LRU Cache Implementation

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
    private var capacity: Int
    private var map = [Int: ListNode]()
    private var head = ListNode(0, 0)
    private var tail = ListNode(0, 0)

    init(_ capacity: Int) {
        self.capacity = capacity
        head.next = tail
        tail.prev = head
    }

    func get(_ key: Int) -> Int {
        if let node = map[key] {
            moveToHead(node)
            return node.value
        }
        return -1
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
                let tail = removeTail()
                map.removeValue(forKey: tail.key)
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
