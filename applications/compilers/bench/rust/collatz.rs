fn steps(mut n: u64) -> u32 {
    let mut count = 0;
    while n != 1 {
        n = if n % 2 == 0 { n / 2 } else { 3 * n + 1 };
        count += 1;
    }
    count
}

fn main() {
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).unwrap();
    let limit = line.trim().parse::<u64>().unwrap();
    let mut best = 1;
    let mut longest = 0;
    for start in 1..limit {
        let s = steps(start);
        if s > longest {
            longest = s;
            best = start;
        }
    }
    println!("longest chain below {} starts at {} with {} steps", limit, best, longest);
}
