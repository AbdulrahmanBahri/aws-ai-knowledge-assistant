#!/usr/bin/env python3

import aws_cdk as cdk

from infrastructure.stacks.storage_stack import StorageStack
from infrastructure.stacks.query_stack import AIKnowledgeAssistantQueryStack

app = cdk.App()

env = cdk.Environment(
    account=app.account,
    region="eu-central-1",
)

StorageStack(
    app,
    "AIKnowledgeAssistantStorageStack",
    env=env,
)

AIKnowledgeAssistantQueryStack(
    app,
    "AIKnowledgeAssistantQueryStack",
    env=env,
)

app.synth()