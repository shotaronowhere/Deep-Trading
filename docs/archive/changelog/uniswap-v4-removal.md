# Uniswap v4 Dependency Removal

## Summary

Uniswap v4 dependencies were removed from this repository because the project no longer uses them.

## Changes

- Removed tracked gitlink path `lib/v4-core`.
- Removed the `lib/v4-core` entry from `foundry.lock`.

## Verification

Use these checks to confirm no v4 dependency references remain:

```bash
rg -n -i --hidden --glob '!.git' 'v4-core|@uniswap/v4|uniswap-v4|uniswap/v4|poolmanager|ipoolmanager|ihooks' .
```

Expected result after this change: no matches.
