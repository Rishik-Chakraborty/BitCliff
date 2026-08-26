---
name: aws-ops
description: AWS conventions for BitCliff. Use whenever creating, modifying, or planning any AWS resource (EC2, S3, IAM, budgets, quotas) or estimating infra cost.
---

# BitCliff AWS Ops

Canon: bitcliff-aws-budget.md. Deviations require explicit approval.

## Fixed decisions
- Region: us-east-1 only. Never create resources elsewhere.
- Auth: IAM user bitcliff-agent (EC2, S3, read-only quotas/budgets). Never touch IAM, never mint keys, never echo secrets into output or commits.
- Prod box: c7i.xlarge, pinned forever once confirmatory rows exist. Never resize; hardware is inside the determinism scope.
- GPU: g6e.xlarge for 0B, g6e.12xlarge for 70B only. Quota: 48 vCPU G-type pending.
- Artifacts: single S3 bucket, public access blocked, 30-day IA lifecycle.
- Site stays Vercel + Cloudflare. No Amplify, Route53, RDS, ECS, or managed services.

## Rules
- Tag every instance transient or persistent. Transient boxes get auto-stop; never leave GPU instances idle.
- Before launching anything: state instance type, estimated $/hr, and run duration.
- Stop and ask before any action over $500 in credits or any cash spend over $50.
- Sync 0B artifacts to S3 before teardown, then terminate, never stop-and-forget.