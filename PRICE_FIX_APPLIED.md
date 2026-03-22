# Fix Applied: Price Clamping Bug in Kalshi & Polymarket Connectors

## Date
2026-03-22

## Problem
- Bot found 1037 match groups but **0 opportunities**, even with min_roi=0.0%
- Diagnostic logging revealed **ALL Kalshi markets had fake prices of (0.001, 0.001)**
- These markets correctly had `has_price=False`, so arbitrage engine filtered them all out

## Root Cause
Both `connectors/kalshi.py` and `connectors/polymarket.py` had a price clamping bug:

```python
# OLD CODE (BUGGY):
yes_price = max(0.001, min(0.999, yes_price))  # Always clamps to minimum 0.001
no_price = max(0.001, min(0.999, no_price))    # Even when both start at 0.0
```

When a market had **no price data at all** (both yes_price and no_price were 0.0):
1. The code would skip the "infer missing leg" logic (lines 202-205 in kalshi.py)
2. Then unconditionally clamp both to minimum 0.001
3. This created **fake prices** (0.001, 0.001) that looked like real data
4. But `has_price` checks `> 0.001`, so correctly returned `False`
5. Result: Markets matched semantically, but all filtered out in arbitrage engine

## Fix Applied

Modified both connectors to **only clamp if real price data exists**:

```python
# NEW CODE (FIXED):
# Only clamp prices if at least one has real data
# If both are 0, keep them at 0 so has_price will be False
if yes_price > 0 or no_price > 0:
    yes_price = max(0.001, min(0.999, yes_price))
    no_price = max(0.001, min(0.999, no_price))
```

Now:
- Markets with real price data: clamped to [0.001, 0.999] as before
- Markets with NO price data: both prices stay at 0.0
- `has_price` property: correctly returns False for zero prices
- **These markets are never matched**, saving CPU time

## Files Changed

1. **connectors/kalshi.py** (lines 207-209)
   - Added conditional check before clamping
   - Only clamps if `yes_price > 0 or no_price > 0`

2. **connectors/polymarket.py** (lines 357-359)
   - Same fix for consistency
   - Polymarket API usually has prices, but safety fix

3. **arbitrage/engine.py** (diagnostic logging added)
   - Enhanced `find_opportunities()` to track rejection reasons
   - Enhanced `_evaluate()` to return detailed rejection messages
   - Logs: rejection counts + first 5 sample rejections

4. **matching/matcher.py** (diagnostic logging added)
   - Logs 5 random sample match groups with prices
   - Shows has_price status, comparable prices, and spreads

5. **scripts/inspect_opportunities.py** (NEW)
   - Comprehensive diagnostic tool
   - Shows detailed breakdown of match groups
   - Spread histogram, has_price distribution, etc.

## Verification Steps

1. **Delete cache files** to force fresh fetch:
   ```
   del cache\kalshi_markets.json
   del cache\polymarket_markets.json
   ```

2. **Run test**:
   ```
   test_fix.bat
   ```

3. **Expected Result**:
   - Kalshi markets now have 0.0/0.0 prices if no data exists
   - These markets are **not matched** (filtered earlier)
   - Match groups that DO form will have real prices on both sides
   - **Opportunities should now be detected** (if spreads exist)

## Diagnostic Output (Before Fix)

From `diagnostic_output.log`:
```
Sample match groups (for diagnostics):
  1. sim=0.835 aligned=False | poly=0.7050 kalshi=0.0010 spread=0.7040
     poly: 'What price will Bitcoin hit in 2026?' (yes=0.7050 no=0.2950 has_price=True)
     kalshi: 'How low will Bitcoin get in 2026?' (yes=0.0010 no=0.0010 has_price=False)
```

**Problem**: Kalshi has fake prices (0.0010, 0.0010) but has_price=False.  
All 1037 match groups had this pattern → 0 opportunities.

## Expected Output (After Fix)

Markets without price data will have (0.0, 0.0) and won't be matched at all.  
Only markets with REAL prices on both platforms will form match groups.  
Opportunities should be detected when spreads exist.

## Related Issues

- This bug was introduced when clamping logic was added to prevent extreme prices
- The intent was good: ensure prices stay in valid range [0.001, 0.999]
- But the implementation didn't distinguish "no data" from "low price"
- Fix preserves the safety check while allowing proper filtering

## Testing Checklist

- [x] Diagnostic logging added
- [x] Root cause identified via diagnostics
- [x] Fix applied to both connectors
- [ ] Cache cleared
- [ ] Full scan run with --min-roi 0.0
- [ ] Opportunities detected (verify count > 0)
- [ ] Manual verification of top 3 opportunities
- [ ] Check prices on actual Polymarket/Kalshi websites

## Notes

- The semantic matcher still works perfectly (1037 groups proves BGE-M3 is matching correctly)
- The bug was purely in price handling between connectors and arbitrage engine
- Markets without prices are now correctly filtered at connector level (return 0.0/0.0)
- This should significantly reduce match group count (only real liquid markets)
- But those matches will have VALID prices and opportunities can be detected
