# Real Dataset Cleaning Example

This example shows how the Python cleaning script handled a real retail sales dataset.

## Cleaning Actions Performed

- normalized transaction dates to YYYY-MM-DD
- standardized boolean-like values in `discount_applied`
- inferred missing `price_per_unit` values from `total_spent ÷ quantity`
- inferred missing `item` values from `(category, price_per_unit)` mapping
- replaced blank `discount_applied` values with `Unknown`

## Result Summary

- Rows before: 12575
- Rows after: 12575
- Duplicates removed: 0
- Missing values significantly reduced in multiple columns

## Remaining Missing Values

Some rows still had missing `total_spent` and `quantity` values that could not be safely inferred. These were left unchanged to avoid incorrect assumptions.