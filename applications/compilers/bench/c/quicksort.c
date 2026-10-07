#include <stdio.h>
#include <stdlib.h>

static void swap(int *a, int *b) { int t = *a; *a = *b; *b = t; }

static void quicksort(int *v, int lo, int hi) {
    if (lo >= hi) return;
    int pivot = v[(lo + hi) / 2], i = lo, j = hi;
    while (i <= j) {
        while (v[i] < pivot) i++;
        while (v[j] > pivot) j--;
        if (i <= j) { swap(&v[i], &v[j]); i++; j--; }
    }
    quicksort(v, lo, j);
    quicksort(v, i, hi);
}

int main(void) {
    int n;
    if (scanf("%d", &n) != 1) return 1;
    int *v = malloc(sizeof(int) * n);
    unsigned state = 7;
    for (int i = 0; i < n; i++) { state = state * 1664525u + 1013904223u; v[i] = (int)(state >> 8) % 1000000; }
    quicksort(v, 0, n - 1);
    long checksum = 0;
    for (int i = 0; i < n; i++) checksum += (long)v[i] * (i % 7 + 1);
    printf("sorted %d, checksum %ld, min %d, max %d\n", n, checksum, v[0], v[n - 1]);
    return 0;
}
