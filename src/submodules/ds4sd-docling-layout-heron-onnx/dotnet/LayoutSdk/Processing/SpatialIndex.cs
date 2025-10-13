using System;
using System.Collections.Generic;
using System.Linq;

namespace LayoutSdk.Processing;

/// <summary>
/// Spatial index for efficient spatial queries on bounding boxes.
/// Uses a simple grid-based approach for fast nearby box lookups.
/// </summary>
public sealed class SpatialIndex<T>
{
    private readonly Dictionary<(int, int), List<SpatialItem>> _grid;
    private readonly float _cellSize;
    private readonly float _invCellSize;

    /// <summary>
    /// Represents an item in the spatial index.
    /// </summary>
    private readonly struct SpatialItem
    {
        public readonly T Item;
        public readonly int Index;
        public readonly float MinX, MinY, MaxX, MaxY;

        public SpatialItem(T item, int index, float minX, float minY, float maxX, float maxY)
        {
            Item = item;
            Index = index;
            MinX = minX;
            MinY = minY;
            MaxX = maxX;
            MaxY = maxY;
        }
    }

    /// <summary>
    /// Initializes a new spatial index.
    /// </summary>
    /// <param name="cellSize">Size of each grid cell</param>
    public SpatialIndex(float cellSize)
    {
        if (cellSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(cellSize), "Cell size must be positive");
        }

        _cellSize = cellSize;
        _invCellSize = 1.0f / cellSize;
        _grid = new Dictionary<(int, int), List<SpatialItem>>();
    }

    /// <summary>
    /// Inserts an item into the spatial index.
    /// </summary>
    /// <param name="item">Item to insert</param>
    /// <param name="index">Index of the item</param>
    public void Insert(T item, int index)
    {
        if (item is not BoundingBox box)
        {
            throw new ArgumentException("Item must be a BoundingBox", nameof(item));
        }

        var spatialItem = new SpatialItem(item, index, box.X, box.Y, box.X + box.Width, box.Y + box.Height);

        // Determina le celle che il box occupa
        int minCellX = (int)Math.Floor(spatialItem.MinX * _invCellSize);
        int minCellY = (int)Math.Floor(spatialItem.MinY * _invCellSize);
        int maxCellX = (int)Math.Floor(spatialItem.MaxX * _invCellSize);
        int maxCellY = (int)Math.Floor(spatialItem.MaxY * _invCellSize);

        // Inserisce nelle celle appropriate
        for (int cellY = minCellY; cellY <= maxCellY; cellY++)
        {
            for (int cellX = minCellX; cellX <= maxCellX; cellX++)
            {
                var cellKey = (cellX, cellY);

                if (!_grid.TryGetValue(cellKey, out var cell))
                {
                    cell = new List<SpatialItem>();
                    _grid[cellKey] = cell;
                }

                cell.Add(spatialItem);
            }
        }
    }

    /// <summary>
    /// Finds all items within the specified radius of the query box.
    /// </summary>
    /// <param name="queryBox">Query bounding box</param>
    /// <param name="radius">Search radius</param>
    /// <returns>Items within the radius</returns>
    public IReadOnlyList<T> GetNearby(BoundingBox queryBox, float radius)
    {
        var result = new HashSet<T>();

        // Determina l'area di ricerca
        float searchMinX = queryBox.X - radius;
        float searchMinY = queryBox.Y - radius;
        float searchMaxX = queryBox.X + queryBox.Width + radius;
        float searchMaxY = queryBox.Y + queryBox.Height + radius;

        int minCellX = (int)Math.Floor(searchMinX * _invCellSize);
        int minCellY = (int)Math.Floor(searchMinY * _invCellSize);
        int maxCellX = (int)Math.Floor(searchMaxX * _invCellSize);
        int maxCellY = (int)Math.Floor(searchMaxY * _invCellSize);

        // Cerca nelle celle rilevanti
        for (int cellY = minCellY; cellY <= maxCellY; cellY++)
        {
            for (int cellX = minCellX; cellX <= maxCellX; cellX++)
            {
                var cellKey = (cellX, cellY);

                if (!_grid.TryGetValue(cellKey, out var cell))
                {
                    continue;
                }

                foreach (var item in cell)
                {
                    if (IsWithinRadius(item, queryBox, radius))
                    {
                        result.Add(item.Item);
                    }
                }
            }
        }

        return result.ToList();
    }

    /// <summary>
    /// Finds all items that intersect with the query box.
    /// </summary>
    /// <param name="queryBox">Query bounding box</param>
    /// <returns>Intersecting items</returns>
    public IReadOnlyList<T> GetIntersecting(BoundingBox queryBox)
    {
        var result = new HashSet<T>();

        // Determina le celle che il query box occupa
        int minCellX = (int)Math.Floor(queryBox.X * _invCellSize);
        int minCellY = (int)Math.Floor(queryBox.Y * _invCellSize);
        int maxCellX = (int)Math.Floor((queryBox.X + queryBox.Width) * _invCellSize);
        int maxCellY = (int)Math.Floor((queryBox.Y + queryBox.Height) * _invCellSize);

        // Cerca nelle celle rilevanti
        for (int cellY = minCellY; cellY <= maxCellY; cellY++)
        {
            for (int cellX = minCellX; cellX <= maxCellX; cellX++)
            {
                var cellKey = (cellX, cellY);

                if (!_grid.TryGetValue(cellKey, out var cell))
                {
                    continue;
                }

                foreach (var item in cell)
                {
                    if (BoxesIntersect(queryBox, item))
                    {
                        result.Add(item.Item);
                    }
                }
            }
        }

        return result.ToList();
    }

    /// <summary>
    /// Gets all items in the spatial index.
    /// </summary>
    /// <returns>All items</returns>
    public IReadOnlyList<T> GetAll()
    {
        return _grid.Values.SelectMany(cell => cell.Select(item => item.Item)).Distinct().ToList();
    }

    /// <summary>
    /// Gets the number of items in the spatial index.
    /// </summary>
    /// <returns>Number of items</returns>
    public int Count()
    {
        return _grid.Values.Sum(cell => cell.Count);
    }

    /// <summary>
    /// Clears the spatial index.
    /// </summary>
    public void Clear()
    {
        _grid.Clear();
    }

    /// <summary>
    /// Checks if a spatial item is within the specified radius of a query box.
    /// </summary>
    private bool IsWithinRadius(SpatialItem item, BoundingBox queryBox, float radius)
    {
        // Calcola distanza tra centri
        float centerX1 = (item.MinX + item.MaxX) / 2f;
        float centerY1 = (item.MinY + item.MaxY) / 2f;
        float centerX2 = queryBox.X + queryBox.Width / 2f;
        float centerY2 = queryBox.Y + queryBox.Height / 2f;

        float dx = centerX1 - centerX2;
        float dy = centerY1 - centerY2;
        float distance = MathF.Sqrt(dx * dx + dy * dy);

        return distance <= radius;
    }

    /// <summary>
    /// Checks if two bounding boxes intersect.
    /// </summary>
    private bool BoxesIntersect(BoundingBox box1, SpatialItem item)
    {
        return !(box1.X >= item.MaxX || box1.X + box1.Width <= item.MinX ||
                box1.Y >= item.MaxY || box1.Y + box1.Height <= item.MinY);
    }
}