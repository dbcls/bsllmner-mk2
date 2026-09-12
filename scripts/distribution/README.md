# Publishing a release

Tooling for distributing bsllmner-mk2 output as a public dataset on AWS S3.

A **release** is one RO-Crate, published two ways: unpacked and browsable under
`releases/<release-id>/`, and as a single `releases/<release-id>.tar.gz` for
bulk download. The reference deployment is `s3://biosampleplus`, served at
<https://biosampleplus.s3.ap-northeast-1.amazonaws.com/>.

## Where this runs

The host you publish from must be able to **both** read the crate and reach the
internet. On shared HPC sites these are not always the same machine: a login
node that mounts the shared filesystem may have no outbound DNS, in which case
`aws` will simply time out even though the crate is plainly visible. Check both
before starting a long transfer:

```bash
ls <crate-dir>/ro-crate-metadata.json
aws sts get-caller-identity
```

The AWS CLI v2 can be installed without root:

```bash
curl -fsSL https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o awscliv2.zip
unzip -q awscliv2.zip && ./aws/install -i "$HOME/local/aws-cli" -b "$HOME/local/bin"
```

## Credentials

Publishing uses a dedicated IAM user (`biosampleplus-uploader` in the reference
deployment) whose policy allows writing **only** to `releases/*` and
`index.json` in the one bucket — not the bucket's `LICENSE`, `README.md`,
`SOURCES.md` or `index.html`, not any other bucket, and nothing else in AWS.

Put its credentials in `~/.aws/credentials` and select the profile:

```bash
export AWS_PROFILE=biosampleplus
aws s3 ls s3://biosampleplus/
```

## Publish

```bash
bash publish-crate.sh <crate-dir> <release-id>
```

Release IDs are `<execution-year>-<month>_<model>`, e.g.
`2026-06_mistral-small3.1-24b`. The ID is permanent and appears in citations.

The script, in order:

1. refuses to overwrite an existing release — pass `--force` only to repair a
   genuinely broken upload, and prefer minting a new ID;
2. verifies the payload against `provenance/checksums.sha256` and aborts on any
   mismatch (`--skip-verify` skips this, but don't);
3. builds `<release-id>.tar.gz` with `pigz -9`;
4. uploads the crate **byte-for-byte**, one pass per file extension so objects
   carry useful `Content-Type` values;
5. uploads the tarball;
6. rewrites `index.json` from `ro-crate-metadata.json`.

### Why the payload is never recompressed

`provenance/checksums.sha256` and every `File` digest in
`ro-crate-metadata.json` describe the *uncompressed* bytes at their *exact*
relative paths. Gzipping individual files, renaming them, or rearranging
directories invalidates the crate and breaks `sha256sum -c`. Bulk downloaders
are served by the `.tar.gz` instead — for this data the archive is roughly 35x
smaller than the unpacked crate, because the result files are highly similar and
gzip compresses across the whole stream.

## Verify after publishing

```bash
curl -s https://biosampleplus.s3.ap-northeast-1.amazonaws.com/index.json | jq .releases

# fetch it back the way a stranger would, with no credentials
curl -O https://biosampleplus.s3.ap-northeast-1.amazonaws.com/releases/<release-id>.tar.gz
tar xzf <release-id>.tar.gz
cd <release-id> && sha256sum -c provenance/checksums.sha256
```

## If the credentials leak

```bash
bash revoke-uploader.sh          # needs an admin profile for the account
```

Takes effect in seconds. Nothing published is lost: the bucket is versioned, and
the uploader cannot touch anything outside `releases/`.
