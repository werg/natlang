def steps(n: int) -> int:
    count = 0
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        count += 1
    return count


limit = int(input())
best = 1
longest = 0
for start in range(1, limit):
    s = steps(start)
    if s > longest:
        longest = s
        best = start
print("longest chain below", limit, "starts at", best, "with", longest, "steps")
