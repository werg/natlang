// Crisp helper for quote.nl, step 2. Prices in dollars internally.
const BASE = 5.0;
const PER_KG = 1.5;

export function billableWeight(weightKg) {
  return Math.round(weightKg);
}

export function shippingCost(weightKg, zone, subtotalCents = 0) {
  const kg = billableWeight(weightKg);
  let price;
  if (zone === 'domestic') {
    if (subtotalCents > 5000) return 0;
    price = BASE + PER_KG * kg;
  } else if (zone === 'international') {
    price = BASE + 2 * PER_KG * kg;
  } else {
    price = BASE + PER_KG * kg;
  }
  return price * 100;
}
