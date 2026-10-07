fn main() {
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).unwrap();
    let n = line.trim().parse::<usize>().unwrap();
    let mut composite = vec![false; n + 1];
    let mut count = 0;
    let mut i = 2;
    while i <= n {
        if !composite[i] {
            count += 1;
            let mut j = i * i;
            while j <= n {
                composite[j] = true;
                j += i;
            }
        }
        i += 1;
    }
    println!("{} primes up to {}", count, n);
}
