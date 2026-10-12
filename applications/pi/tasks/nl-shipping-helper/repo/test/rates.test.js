import { test } from 'node:test';
import assert from 'node:assert/strict';
import { shippingCost } from '../helpers/rates.js';

test('domestic, whole kilograms', () => assert.equal(shippingCost(3, 'domestic', 1000), 950));
