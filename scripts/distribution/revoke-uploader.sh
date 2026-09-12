#!/usr/bin/env bash
# Revoke every access key belonging to biosampleplus-uploader.
# Run with an admin profile for account 928810569478.
set -euo pipefail
PROFILE="${AWS_PROFILE:-dbcls}"
USER=biosampleplus-uploader
keys=$(aws iam list-access-keys --user-name "$USER" --profile "$PROFILE" \
        --query 'AccessKeyMetadata[].AccessKeyId' --output text)
if [ -z "$keys" ]; then echo "no access keys on $USER"; exit 0; fi
for k in $keys; do
  aws iam delete-access-key --user-name "$USER" --access-key-id "$k" --profile "$PROFILE"
  echo "revoked $k"
done
echo "done. issue a new key with:"
echo "  aws iam create-access-key --user-name $USER --profile $PROFILE"
