#include <stdio.h>

struct point { int x, y; };
struct box { struct point low, high; };

static struct box bound(const struct point *p, int n) {
    struct box b = { p[0], p[0] };
    for (int i = 1; i < n; i++) {
        if (p[i].x < b.low.x) b.low.x = p[i].x;
        if (p[i].y < b.low.y) b.low.y = p[i].y;
        if (p[i].x > b.high.x) b.high.x = p[i].x;
        if (p[i].y > b.high.y) b.high.y = p[i].y;
    }
    return b;
}

static struct point points[100000];

int main(void) {
    int n, rounds;
    if (scanf("%d %d", &n, &rounds) != 2) return 1;
    long area = 0;
    for (int r = 0; r < rounds; r++) {
        for (int i = 0; i < n; i++) { points[i].x = (i * 7919 + r * 104729) % 10007 - 5000; points[i].y = (i * 6271 + r * 7) % 9973 - 4000; }
        struct box b = bound(points, n);
        area += (long)(b.high.x - b.low.x) * (b.high.y - b.low.y);
    }
    printf("total area %ld\n", area);
    return 0;
}
