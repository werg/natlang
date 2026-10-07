struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn dist(&self, other: &Point) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

fn main() {
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).unwrap();
    let mut words = line.split_whitespace();
    let n = words.next().unwrap().parse::<u64>().unwrap();
    let rounds = words.next().unwrap().parse::<u64>().unwrap();
    let origin = Point { x: 0.0, y: 0.0 };
    let mut total = 0.0;
    for r in 0..rounds {
        for i in 0..n {
            let p = Point { x: ((i * 7919 + r * 104729) % 10007) as f64 - 5000.0, y: ((i * 6271 + r * 7) % 9973) as f64 - 4000.0 };
            total += p.dist(&origin);
        }
    }
    println!("mean distance {:.6}", total / (n * rounds) as f64);
}
