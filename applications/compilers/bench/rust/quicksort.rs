struct Lcg {
    state: u64,
}

impl Lcg {
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state >> 33
    }
}

/// Sorts v[lo..hi].
fn quicksort(v: &mut Vec<u64>, lo: usize, hi: usize) {
    if lo + 1 >= hi {
        return;
    }
    let pivot = v[hi - 1];
    let mut store = lo;
    for i in lo..hi - 1 {
        if v[i] < pivot {
            v.swap(i, store);
            store += 1;
        }
    }
    v.swap(store, hi - 1);
    quicksort(v, lo, store);
    quicksort(v, store + 1, hi);
}

fn main() {
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).unwrap();
    let n = line.trim().parse::<usize>().unwrap();
    let mut rng = Lcg { state: 42 };
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(rng.next() % 1000000);
    }
    quicksort(&mut v, 0, n);
    let mut checksum: u64 = 0;
    for i in 0..n {
        checksum = checksum.wrapping_mul(31).wrapping_add(v[i]);
    }
    println!("sorted {} values: first {}, last {}, checksum {}", n, v[0], v[n - 1], checksum);
}
