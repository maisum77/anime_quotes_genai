# IAM Policies - Least Privilege Configuration

This document outlines the minimum IAM permissions required for each component in the Anime Quote Generator pipeline.

## Overview

The system follows the principle of least privilege, where each Lambda function only has permissions to perform its specific tasks. Policies are scoped to specific resources and actions.

## Policy Categories

### 1. Preprocessing Lambda Policy

**Function**: `anime-quote-preprocessing`
**Purpose**: Validate input, generate job IDs, forward to SQS

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "SQSWritePermissions",
      "Effect": "Allow",
      "Action": [
        "sqs:SendMessage",
        "sqs:GetQueueAttributes",
        "sqs:GetQueueUrl"
      ],
      "Resource": [
        "arn:aws:sqs:*:*:generation-queue",
        "arn:aws:sqs:*:*:generation-dlq"
      ]
    },
    {
      "Sid": "S3WriteInputs",
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:GetObject"],
      "Resource": [
        "arn:aws:s3:::anime-quote-generator/inputs/*",
        "arn:aws:s3:::anime-quote-generator/logs/*"
      ]
    },
    {
      "Sid": "DynamoDBJobTracking",
      "Effect": "Allow",
      "Action": ["dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:GetItem"],
      "Resource": "arn:aws:dynamodb:*:*:table/GenerationJobs"
    },
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:log-group:/aws/lambda/anime-quote-preprocessing:*"
    }
  ]
}
```

### 2. Generation Lambda Policy

**Function**: `anime-quote-generation`
**Purpose**: Generate content using Gemini/GPT-2/fallback

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "SQSPermissions",
      "Effect": "Allow",
      "Action": [
        "sqs:ReceiveMessage",
        "sqs:DeleteMessage",
        "sqs:GetQueueAttributes",
        "sqs:GetQueueUrl",
        "sqs:SendMessage"
      ],
      "Resource": [
        "arn:aws:sqs:*:*:generation-queue",
        "arn:aws:sqs:*:*:postprocessing-queue",
        "arn:aws:sqs:*:*:generation-dlq"
      ]
    },
    {
      "Sid": "DynamoDBJobUpdates",
      "Effect": "Allow",
      "Action": ["dynamodb:UpdateItem", "dynamodb:GetItem"],
      "Resource": "arn:aws:dynamodb:*:*:table/GenerationJobs"
    },
    {
      "Sid": "S3ModelAccess",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::anime-quote-generator",
        "arn:aws:s3:::anime-quote-generator/models/*"
      ]
    },
    {
      "Sid": "CloudWatchMetrics",
      "Effect": "Allow",
      "Action": ["cloudwatch:PutMetricData"],
      "Resource": "*"
    },
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:log-group:/aws/lambda/anime-quote-generation:*"
    },
    {
      "Sid": "SecretsManagerAccess",
      "Effect": "Allow",
      "Action": ["secretsmanager:GetSecretValue"],
      "Resource": "arn:aws:secretsmanager:*:*:secret:GOOGLE_API_KEY-*"
    }
  ]
}
```

### 3. Postprocessing Lambda Policy

**Function**: `anime-quote-postprocessing`
**Purpose**: Format output, store in S3, send notifications

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "SQSPermissions",
      "Effect": "Allow",
      "Action": [
        "sqs:ReceiveMessage",
        "sqs:DeleteMessage",
        "sqs:GetQueueAttributes",
        "sqs:GetQueueUrl"
      ],
      "Resource": [
        "arn:aws:sqs:*:*:postprocessing-queue",
        "arn:aws:sqs:*:*:notification-queue"
      ]
    },
    {
      "Sid": "S3OutputStorage",
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::anime-quote-generator",
        "arn:aws:s3:::anime-quote-generator/outputs/*",
        "arn:aws:s3:::anime-quote-generator/logs/*"
      ]
    },
    {
      "Sid": "DynamoDBPermissions",
      "Effect": "Allow",
      "Action": [
        "dynamodb:UpdateItem",
        "dynamodb:PutItem",
        "dynamodb:GetItem",
        "dynamodb:Query"
      ],
      "Resource": [
        "arn:aws:dynamodb:*:*:table/GenerationJobs",
        "arn:aws:dynamodb:*:*:table/UserHistory"
      ]
    },
    {
      "Sid": "SNSPermissions",
      "Effect": "Allow",
      "Action": ["sns:Publish"],
      "Resource": [
        "arn:aws:sns:*:*:generation-complete",
        "arn:aws:sns:*:*:generation-failed"
      ]
    },
    {
      "Sid": "CloudWatchMetrics",
      "Effect": "Allow",
      "Action": ["cloudwatch:PutMetricData"],
      "Resource": "*"
    },
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:log-group:/aws/lambda/anime-quote-postprocessing:*"
    }
  ]
}
```

### 4. API Gateway Execution Role

**Role**: `api-gateway-execution-role`
**Purpose**: Allow API Gateway to invoke Lambda functions

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "LambdaInvokePermissions",
      "Effect": "Allow",
      "Action": ["lambda:InvokeFunction"],
      "Resource": [
        "arn:aws:lambda:*:*:function:anime-quote-preprocessing",
        "arn:aws:lambda:*:*:function:anime-quote-generation",
        "arn:aws:lambda:*:*:function:anime-quote-postprocessing"
      ]
    },
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:log-group:/aws/apigateway/*"
    }
  ]
}
```

### 5. EventBridge/Scheduler Role

**Role**: `eventbridge-scheduler-role`
**Purpose**: Schedule batch jobs and trigger maintenance tasks

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "LambdaInvokePermissions",
      "Effect": "Allow",
      "Action": ["lambda:InvokeFunction"],
      "Resource": [
        "arn:aws:lambda:*:*:function:anime-quote-preprocessing",
        "arn:aws:lambda:*:*:function:anime-quote-batch-orchestrator"
      ]
    },
    {
      "Sid": "SQSPermissions",
      "Effect": "Allow",
      "Action": ["sqs:SendMessage"],
      "Resource": "arn:aws:sqs:*:*:generation-queue"
    }
  ]
}
```

## Resource Naming Convention

All resources follow a consistent naming pattern:

- S3 Bucket: `anime-quote-generator-{environment}` (e.g., `anime-quote-generator-prod`)
- SQS Queues: `{function}-queue`, `{function}-dlq`
- SNS Topics: `{event}-topic`
- DynamoDB Tables: `{EntityName}{Environment}` (e.g., `GenerationJobsProd`)
- Lambda Functions: `anime-quote-{function-name}`

## Environment-Specific Policies

For multi-environment deployments (dev, staging, prod), policies should include environment tags:

```json
{
  "Condition": {
    "StringEquals": {
      "aws:ResourceTag/Environment": "production"
    }
  }
}
```

## Security Best Practices Implemented

1. **Principle of Least Privilege**: Each function only has permissions for its specific tasks
2. **Resource-Level Permissions**: Policies specify exact ARNs, not wildcards (`*`) where possible
3. **Condition Keys**: Use conditions to restrict access based on tags, source IP, etc.
4. **No Administrative Permissions**: No functions have `*:*` permissions
5. **Separate Roles**: Each Lambda function has its own execution role
6. **Encryption Enforcement**: All S3 buckets require SSE-S3 or SSE-KMS encryption
7. **VPC Isolation**: Generation Lambda (with GPT-2) runs in VPC for model security
8. **Secret Management**: API keys stored in AWS Secrets Manager, not environment variables
9. **Audit Trail**: All actions logged to CloudTrail
10. **Regular Rotation**: IAM roles and policies reviewed quarterly

## CDK Implementation Example

```python
from aws_cdk import (
    aws_iam as iam,
    aws_lambda as lambda_,
)

# Preprocessing Lambda role
preprocessing_role = iam.Role(
    self, "PreprocessingRole",
    assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
    description="Execution role for preprocessing Lambda"
)

# Attach managed policy for basic Lambda execution
preprocessing_role.add_managed_policy(
    iam.ManagedPolicy.from_aws_managed_policy_name(
        "service-role/AWSLambdaBasicExecutionRole"
    )
)

# Add custom inline policy
preprocessing_role.add_to_policy(
    iam.PolicyStatement(
        effect=iam.Effect.ALLOW,
        actions=[
            "sqs:SendMessage",
            "sqs:GetQueueAttributes",
        ],
        resources=[generation_queue.queue_arn]
    )
)

# Create Lambda with the role
preprocessing_lambda = lambda_.Function(
    self, "PreprocessingFunction",
    runtime=lambda_.Runtime.PYTHON_3_12,
    handler="preprocessing.lambda_handler",
    code=lambda_.Code.from_asset("./lambda"),
    role=preprocessing_role,
    environment={
        "GENERATION_QUEUE_URL": generation_queue.queue_url,
        "JOBS_TABLE": jobs_table.table_name,
    }
)
```

## Monitoring and Compliance

- Use AWS Config to monitor IAM policy compliance
- Enable AWS Organizations SCPs to enforce security boundaries
- Regular security audits using AWS Security Hub
- IAM Access Analyzer for policy validation
- CloudTrail logs for all IAM actions
