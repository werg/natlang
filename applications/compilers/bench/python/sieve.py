def count_primes(limit: int) -> int:
    composite: list[int] = []
    for i in range(limit + 1):
        composite.append(0)
    count = 0
    for i in range(2, limit + 1):
        if composite[i] == 0:
            count += 1
            for j in range(i * i, limit + 1, i):
                composite[j] = 1
    return count


limit = int(input())
print(count_primes(limit), "primes up to", limit)
