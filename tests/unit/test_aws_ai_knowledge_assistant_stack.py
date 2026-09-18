import aws_cdk as core
import aws_cdk.assertions as assertions

from aws_ai_knowledge_assistant.aws_ai_knowledge_assistant_stack import AwsAiKnowledgeAssistantStack

# example tests. To run these tests, uncomment this file along with the example
# resource in aws_ai_knowledge_assistant/aws_ai_knowledge_assistant_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = AwsAiKnowledgeAssistantStack(app, "aws-ai-knowledge-assistant")
    template = assertions.Template.from_stack(stack)

#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })
