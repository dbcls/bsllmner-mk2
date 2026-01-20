# Architecture

## Batch Processing

バッチ処理ループは `cli_common.py` の `process_batches()` に共通化。Extract/Select 両モードで使用する。

```python
await process_batches(
    entries=bs_entries,
    batch_size=args.batch_size,
    process_fn=process_batch_fn,
    on_batch_complete=on_complete_callback,
)
```
