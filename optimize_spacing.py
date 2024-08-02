def find_factor_pairs(goal):
    factor_pairs = []

    for i in range(1, int(goal**0.5) + 1):
        if goal % i == 0:
            factor_pairs.append([i, goal // i])
    
    return factor_pairs

# Example usage
goal = 16
combinations = find_factor_pairs(goal)
print(combinations)