# Economic game

Natlang chooses merchant buy/pass intents from private holdings and public
offers. The seeded host settles submitted intents in a stable round order,
checks cash and stock, and asserts conservation of money and goods. Model
policy randomness and world ordering are separate; this fixture uses the
explicit world seed for order, and a caller supplies the interpreter seed.
