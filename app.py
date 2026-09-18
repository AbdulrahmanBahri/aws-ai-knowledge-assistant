#!/usr/bin/env python3

import aws_cdk as cdk

from infrastructure.stacks.storage_stack import StorageStack


app = cdk.App()


StorageStack(
    app,
    "AIKnowledgeAssistantStorageStack",
    env=cdk.Environment(
        account=app.account,
        region="eu-central-1",
    ),
)


app.synth()