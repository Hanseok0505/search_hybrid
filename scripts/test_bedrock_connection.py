from __future__ import annotations

import json
import os

import boto3
from botocore.exceptions import BotoCoreError, ClientError


def main() -> None:
    region = os.getenv("BEDROCK_REGION", "us-east-1")
    model_id = os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v1")
    should_invoke = os.getenv("BEDROCK_INVOKE_TEST", "false").lower() == "true"

    session = boto3.Session(region_name=region)
    creds = session.get_credentials()
    if creds is None:
        print("NO_AWS_CREDENTIALS")
        return

    print(f"BEDROCK_REGION={region}")
    print("AWS_CREDENTIALS=OK")

    if not should_invoke:
        print("SKIP_INVOKE (set BEDROCK_INVOKE_TEST=true to run model invocation)")
        return

    client = session.client("bedrock-runtime")
    body = {"inputText": "bedrock embedding test"}
    try:
        res = client.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        parsed = json.loads(res["body"].read().decode("utf-8"))
        emb = parsed.get("embedding") or parsed.get("embeddings", [[]])[0]
        print(f"INVOKE_OK dim={len(emb)} model={model_id}")
    except (BotoCoreError, ClientError) as exc:
        print(f"INVOKE_FAILED {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
