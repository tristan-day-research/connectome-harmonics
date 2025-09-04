# Custom Linter Rules

## Rule: No Settings Loading in Helper Functions

### Pattern to Flag:
```python
def helper_function(settings=None):
    if settings is None:
        from ch.settings import load_settings  # ❌ FLAG THIS
        settings = load_settings()
```

### Correct Pattern:
```python
def helper_function(data_path: Union[str, Path]):
    """Helper function with explicit path parameter."""
    return pd.read_parquet(data_path)
```

### Why This Matters:
- Hidden dependencies make functions hard to test
- Tight coupling reduces flexibility
- Settings should be loaded at application level
- Functions should be pure and predictable

### Manual Check:
Before committing, search for:
- `from ch.settings import load_settings` inside function bodies
- `if settings is None:` patterns in utility functions
- Functions that take `settings=None` as first parameter

### Fix:
1. Change function to take explicit path parameter
2. Load settings at call site
3. Pass path to function
4. Update function documentation
