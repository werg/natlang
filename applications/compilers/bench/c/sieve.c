#include <stdio.h>
#include <stdlib.h>

int count_primes(int limit) {
    char *composite = calloc(limit + 1, 1);
    int count = 0;
    for (int i = 2; i <= limit; i++) {
        if (composite[i]) continue;
        count++;
        for (long j = (long)i * i; j <= limit; j += i) composite[j] = 1;
    }
    free(composite);
    return count;
}

int main(void) {
    int limit;
    if (scanf("%d", &limit) != 1) return 1;
    printf("%d primes up to %d\n", count_primes(limit), limit);
    return 0;
}
