# Memo

## Resume 機能の設計

### Select モードの orphan 処理

Select モードの Resume で orphan (Extract完了・Select未完了) が発生した場合:
- 警告ログを出力
- 再処理対象に含める

不正データ (Select にあって Extract にない) は `ResumeDataError` を raise する。

### Resume ファイルの整合性保証

Select モードでは `on_batch_complete` で Extract と Select の resume ファイルを**同時に保存**することで整合性を保証する。
