"""First Task"""
import argparse
import sys


def count_islands(grid):
    """Counting islands. Classical Algorithms"""
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if any(len(row) != cols for row in grid):
        raise ValueError("Grid must be rectangular")
    if any(value not in (0, 1) for row in grid for value in row):
        raise ValueError("Grid must contain only 0 and 1")
    visited = set()
    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 1 or (r, c) in visited:
                continue
            count += 1
            visited.add((r, c))
            stack = [(r, c)]
            while stack:
                x, y = stack.pop()
                for nx, ny in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
                    if (0 <= nx < rows and 0 <= ny < cols
                            and grid[nx][ny] == 1 and (nx, ny) not in visited):
                        visited.add((nx, ny))
                        stack.append((nx, ny))
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        tokens = [int(x) for x in sys.stdin.read().split()]
        if len(tokens) < 2:
            raise ValueError("Expected M N followed by M*N values")
        m, n = tokens[:2]
        if m < 0 or n < 0 or len(tokens) != 2 + m*n:
            raise ValueError("Invalid dimensions or cell count")
        grid = [tokens[2+r*n:2+(r+1)*n] for r in range(m)]
        print(count_islands(grid))
    except ValueError as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
