using System;
using System.Collections.Generic;

namespace LayoutSdk.Processing;

/// <summary>
/// Union-Find (Disjoint Set Union) data structure optimized for layout post-processing.
/// Uses path compression and union by rank for optimal performance.
/// </summary>
public sealed class UnionFind<T>
{
    private readonly int[] _parent;
    private readonly int[] _rank;
    private readonly T[] _items;
    private readonly Dictionary<int, List<T>> _groups;

    /// <summary>
    /// Initializes a new UnionFind structure.
    /// </summary>
    /// <param name="items">Items to track unions for</param>
    public UnionFind(IReadOnlyList<T> items)
    {
        if (items == null)
        {
            throw new ArgumentNullException(nameof(items));
        }

        int count = items.Count;
        _parent = new int[count];
        _rank = new int[count];
        _items = new T[count];
        _groups = new Dictionary<int, List<T>>();

        for (int i = 0; i < count; i++)
        {
            _parent[i] = i;
            _rank[i] = 0;
            _items[i] = items[i];
            _groups[i] = new List<T> { items[i] };
        }
    }

    /// <summary>
    /// Finds the root of the set containing the specified item.
    /// </summary>
    /// <param name="index">Index of the item</param>
    /// <returns>Root index of the set</returns>
    public int Find(int index)
    {
        if (index < 0 || index >= _parent.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(index));
        }

        // Path compression
        if (_parent[index] != index)
        {
            _parent[index] = Find(_parent[index]);
        }

        return _parent[index];
    }

    /// <summary>
    /// Unions two sets containing the specified items.
    /// </summary>
    /// <param name="index1">Index of first item</param>
    /// <param name="index2">Index of second item</param>
    public void Union(int index1, int index2)
    {
        if (index1 < 0 || index1 >= _parent.Length || index2 < 0 || index2 >= _parent.Length)
        {
            throw new ArgumentOutOfRangeException();
        }

        int root1 = Find(index1);
        int root2 = Find(index2);

        if (root1 == root2)
        {
            return; // Already in same set
        }

        // Union by rank
        if (_rank[root1] < _rank[root2])
        {
            _parent[root1] = root2;
            _groups[root2].AddRange(_groups[root1]);
            _groups.Remove(root1);
        }
        else if (_rank[root1] > _rank[root2])
        {
            _parent[root2] = root1;
            _groups[root1].AddRange(_groups[root2]);
            _groups.Remove(root2);
        }
        else
        {
            _parent[root2] = root1;
            _rank[root1]++;
            _groups[root1].AddRange(_groups[root2]);
            _groups.Remove(root2);
        }
    }

    /// <summary>
    /// Gets all items in the same set as the specified item.
    /// </summary>
    /// <param name="index">Index of the item</param>
    /// <returns>List of items in the same set</returns>
    public IReadOnlyList<T> GetGroupItems(int index)
    {
        int root = Find(index);
        return _groups[root];
    }

    /// <summary>
    /// Gets all items in the set with the specified root.
    /// </summary>
    /// <param name="rootIndex">Root index of the set</param>
    /// <returns>List of items in the set</returns>
    public IReadOnlyList<T> GetGroupItemsByRoot(int rootIndex)
    {
        if (!_groups.TryGetValue(rootIndex, out var group))
        {
            throw new ArgumentException("Invalid root index", nameof(rootIndex));
        }

        return group;
    }

    /// <summary>
    /// Gets all items in the same set as the specified item.
    /// </summary>
    /// <param name="index">Index of the item</param>
    /// <returns>List of items in the same set</returns>
    public IReadOnlyList<T> GetGroupBoxes(int index)
    {
        int root = Find(index);
        return _groups[root];
    }

    /// <summary>
    /// Gets the number of distinct sets.
    /// </summary>
    public int GetSetCount()
    {
        return _groups.Count;
    }

    /// <summary>
    /// Gets all root indices (one per set).
    /// </summary>
    /// <returns>Array of root indices</returns>
    public int[] GetRoots()
    {
        return _groups.Keys.ToArray();
    }

    /// <summary>
    /// Gets the size of the set containing the specified item.
    /// </summary>
    /// <param name="index">Index of the item</param>
    /// <returns>Size of the set</returns>
    public int GetSetSize(int index)
    {
        int root = Find(index);
        return _groups[root].Count;
    }
}