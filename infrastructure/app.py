#!/usr/bin/env python3
"""
CDK App entry point for the Anime Quote Generator infrastructure.

Usage:
    cdk synth                          # Synthesize CloudFormation template
    cdk deploy --context env=dev       # Deploy to development
    cdk deploy --context env=prod      # Deploy to production
    cdk diff --context env=dev         # Compare with deployed stack
"""

import os

import aws_cdk as cdk
from anime_quote_generator.stack import AnimeQuoteGeneratorStack

app = cdk.App()

# Deploy based on context environment
env_name = app.node.try_get_context("env") or "dev"
env_config = app.node.try_get_context(env_name) or {}

# AWS environment (uses CLI credentials if not specified)
env = cdk.Environment(
    account=os.getenv("CDK_DEFAULT_ACCOUNT"),
    region=env_config.get("region", os.getenv("CDK_DEFAULT_REGION", "us-east-1")),
)

stack_name = f"AnimeQuoteGenerator-{env_name.title()}"

AnimeQuoteGeneratorStack(
    app,
    stack_name,
    env_name=env_name,
    env_config=env_config,
    env=env,
    description=f"Anime Quote Generator - {env_name.title()} Environment",
    tags={
        "Project": "AnimeQuoteGenerator",
        "Environment": env_name,
        "ManagedBy": "CDK",
    },
)

app.synth()
