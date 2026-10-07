#include <stdio.h>
#include <stdlib.h>

static void fill(double *m, int n, int seed) {
    unsigned state = seed;
    for (int i = 0; i < n * n; i++) {
        state = state * 1103515245u + 12345u;
        m[i] = (double)((state >> 16) % 100) / 10.0;
    }
}

static void multiply(const double *a, const double *b, double *c, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) sum += a[i * n + k] * b[k * n + j];
            c[i * n + j] = sum;
        }
}

int main(void) {
    int n;
    if (scanf("%d", &n) != 1) return 1;
    double *a = malloc(sizeof(double) * n * n), *b = malloc(sizeof(double) * n * n), *c = malloc(sizeof(double) * n * n);
    fill(a, n, 1);
    fill(b, n, 2);
    multiply(a, b, c, n);
    double trace = 0.0;
    for (int i = 0; i < n; i++) trace += c[i * n + i];
    printf("trace %.1f\n", trace);
    return 0;
}
