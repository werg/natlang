import isNumber from 'is-number';

/** Handwritten check using the application's declared npm dependency. */
export default function is_numeric(value: string): boolean {
  return isNumber(value);
}
