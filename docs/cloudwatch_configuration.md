# CloudWatch Logging, Metrics, and Alarms Configuration

## Overview

This document outlines the comprehensive CloudWatch monitoring setup for the Anime Quote Generator Lambda processing pipeline. The monitoring system includes structured logging, custom metrics, and automated alarms for operational excellence.

## 1. CloudWatch Logging Strategy

### 1.1 Log Groups and Retention

Each Lambda function has its own CloudWatch Log Group with appropriate retention policies:

| Function       | Log Group Name                           | Retention Period | Description                                  |
| -------------- | ---------------------------------------- | ---------------- | -------------------------------------------- |
| Preprocessing  | `/aws/lambda/anime-quote-preprocessing`  | 30 days          | Input validation and request processing logs |
| Generation     | `/aws/lambda/anime-quote-generation`     | 30 days          | AI generation and processing logs            |
| Postprocessing | `/aws/lambda/anime-quote-postprocessing` | 30 days          | Output formatting and storage logs           |
| API Gateway    | `/aws/apigateway/anime-quote-api`        | 90 days          | API request/response logs                    |
| SQS Processing | Custom metrics                           | 30 days          | Queue depth and processing metrics           |

### 1.2 Structured Log Format

All Lambda functions use a consistent JSON log format for machine parsing:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "function": "anime-quote-generation",
  "request_id": "abc123-def456",
  "job_id": "job-20240115-103000",
  "operation": "generate_with_gemini",
  "duration_ms": 1250,
  "metrics": {
    "generation_type": "speech",
    "speech_type": "motivational",
    "characters": ["Naruto", "Sasuke"]
  },
  "message": "Successfully generated speech"
}
```

### 1.3 Log Levels Configuration

- **DEBUG**: Detailed tracing for development and troubleshooting
- **INFO**: Normal operational events, request processing
- **WARNING**: Non-critical issues, performance warnings
- **ERROR**: Processing failures, external service errors
- **CRITICAL**: System failures, security incidents

## 2. Custom Metrics

### 2.1 Metric Namespace

All custom metrics use the namespace: `AnimeQuoteGenerator`

### 2.2 Key Performance Metrics

#### 2.2.1 Processing Metrics

| Metric Name     | Unit    | Dimensions              | Description              | Alarm Threshold |
| --------------- | ------- | ----------------------- | ------------------------ | --------------- |
| `RequestCount`  | Count   | `Function`, `Type`      | Total requests processed | N/A             |
| `SuccessCount`  | Count   | `Function`, `Type`      | Successful requests      | N/A             |
| `ErrorCount`    | Count   | `Function`, `ErrorType` | Failed requests          | >5% of requests |
| `ExecutionTime` | Seconds | `Function`, `Operation` | Function execution time  | >10 seconds     |
| `QueueDepth`    | Count   | `QueueName`             | SQS message backlog      | >100 messages   |

#### 2.2.2 Generation Metrics

| Metric Name         | Unit    | Dimensions       | Description             | Alarm Threshold |
| ------------------- | ------- | ---------------- | ----------------------- | --------------- |
| `GenerationTime`    | Seconds | `Model`, `Type`  | AI generation time      | >30 seconds     |
| `GeminiSuccessRate` | Percent | `GenerationType` | Gemini API success rate | <95%            |
| `GPT2FallbackRate`  | Percent | `SpeechType`     | GPT-2 fallback usage    | >20%            |
| `FallbackUsage`     | Count   | `Reason`         | Fallback quote usage    | N/A             |
| `TokenCount`        | Count   | `Model`          | Generated tokens count  | N/A             |

#### 2.2.3 Resource Metrics

| Metric Name            | Unit         | Dimensions | Description                  | Alarm Threshold |
| ---------------------- | ------------ | ---------- | ---------------------------- | --------------- |
| `MemoryUsage`          | Percent      | `Function` | Lambda memory utilization    | >80%            |
| `DurationBilled`       | Milliseconds | `Function` | Billed duration              | >5000ms         |
| `ColdStarts`           | Count        | `Function` | Lambda cold starts           | >10/hour        |
| `ConcurrentExecutions` | Count        | N/A        | Concurrent Lambda executions | >100            |

### 2.3 Metric Collection Implementation

Metrics are collected using the `log_metric()` function from `lambda/shared/logging.py`:

```python
from shared.logging import log_metric

# Example: Log execution time
log_metric(
    "ExecutionTime",
    execution_time,
    unit="Seconds",
    dimensions={"Function": "generation", "Operation": "gemini_generation"}
)

# Example: Log error
log_metric(
    "ErrorCount",
    1,
    unit="Count",
    dimensions={"Function": "preprocessing", "ErrorType": "ValidationError"}
)
```

## 3. CloudWatch Alarms

### 3.1 Critical Alarms

#### 3.1.1 Error Rate Alarms

```yaml
Alarm: HighErrorRate
  Metric: ErrorCount / RequestCount * 100
  Threshold: >5%
  Period: 5 minutes
  Evaluation Periods: 2
  Statistic: Average
  Action: SNS notification to DevOps team
```

#### 3.1.2 Performance Alarms

```yaml
Alarm: SlowGeneration
  Metric: GenerationTime
  Threshold: >30 seconds
  Period: 1 minute
  Evaluation Periods: 3
  Statistic: p95
  Action: SNS notification, auto-scale investigation
```

#### 3.1.3 Resource Alarms

```yaml
Alarm: HighMemoryUsage
  Metric: MemoryUsage
  Threshold: >80%
  Period: 1 minute
  Evaluation Periods: 2
  Statistic: Maximum
  Action: SNS notification, memory increase recommendation
```

### 3.2 Warning Alarms

#### 3.2.1 Queue Depth Alarms

```yaml
Alarm: HighQueueDepth
  Metric: QueueDepth
  Threshold: >50 messages
  Period: 5 minutes
  Evaluation Periods: 2
  Statistic: Maximum
  Action: SNS notification to monitoring channel
```

#### 3.2.2 Fallback Rate Alarms

```yaml
Alarm: HighFallbackRate
  Metric: GPT2FallbackRate
  Threshold: >20%
  Period: 15 minutes
  Evaluation Periods: 2
  Statistic: Average
  Action: SNS notification, Gemini service check
```

### 3.3 Alarm Actions

All alarms are configured with multiple action targets:

1. **Primary**: SNS topic `anime-quote-alarms` for Slack/Email notifications
2. **Secondary**: Auto-remediation Lambda functions for common issues
3. **Tertiary**: PagerDuty integration for critical alerts (P1/P2)

## 4. CloudWatch Dashboards

### 4.1 Operational Dashboard

**Dashboard Name**: `AnimeQuoteGenerator-Operations`

**Widgets**:

1. **Request Summary**: Total requests, success rate, error rate
2. **Performance Overview**: Average execution times per function
3. **Generation Metrics**: Gemini vs GPT-2 usage, fallback rates
4. **Queue Monitoring**: SQS queue depths and processing rates
5. **Error Breakdown**: Error types and frequencies
6. **Resource Utilization**: Memory, duration, concurrent executions

### 4.2 Business Dashboard

**Dashboard Name**: `AnimeQuoteGenerator-Business`

**Widgets**:

1. **Usage Trends**: Requests per hour/day/week
2. **Popular Content**: Most requested speech types and characters
3. **Generation Quality**: Success rates by generation type
4. **Cost Metrics**: Lambda invocation costs and S3 storage costs
5. **User Satisfaction**: Response time percentiles (p50, p90, p99)

### 4.3 Technical Dashboard

**Dashboard Name**: `AnimeQuoteGenerator-Technical`

**Widgets**:

1. **Lambda Metrics**: Cold starts, memory usage, duration billed
2. **External Services**: Gemini API latency and success rates
3. **Storage Metrics**: S3 object counts and sizes
4. **Database Metrics**: DynamoDB read/write capacity
5. **Network Metrics**: API Gateway latency and 4xx/5xx errors

## 5. Log Insights Queries

### 5.1 Common Troubleshooting Queries

#### 5.1.1 Error Analysis

```sql
-- Find all errors in the last hour
fields @timestamp, @message, @logStream
| filter @message like /ERROR/
| sort @timestamp desc
| limit 100
```

#### 5.1.2 Performance Analysis

```sql
-- Find slow requests (>5 seconds)
fields @timestamp, function, operation, duration_ms
| parse @message '"duration_ms": *,' as duration_ms
| filter duration_ms > 5000
| sort duration_ms desc
```

#### 5.1.3 Usage Patterns

```sql
-- Count requests by generation type
stats count(*) as request_count by generation_type
| filter @message like /processing request/
| sort request_count desc
```

### 5.2 Operational Queries

#### 5.2.1 Cold Start Analysis

```sql
-- Identify cold starts
fields @timestamp, @message
| filter @message like /REPORT/
| filter @message like /Init Duration/
| sort @timestamp desc
```

#### 5.2.2 Cost Optimization

```sql
-- Find high memory usage patterns
fields @timestamp, function, memory_used_mb, memory_limit_mb
| parse @message '"memory_used_mb": *,' as memory_used_mb
| parse @message '"memory_limit_mb": *,' as memory_limit_mb
| display (memory_used_mb / memory_limit_mb) * 100 as memory_percent
| filter memory_percent > 70
```

## 6. Implementation in Lambda Functions

### 6.1 Preprocessing Function Metrics

```python
# In preprocessing.py
from shared.logging import log_metric, log_execution_time

def handle_single_generation(event, context):
    start_time = time.time()

    # Process request
    # ...

    # Log metrics
    log_metric("RequestCount", 1, dimensions={"Function": "preprocessing", "Type": "single"})
    log_execution_time(start_time, "preprocessing_single")

    if success:
        log_metric("SuccessCount", 1, dimensions={"Function": "preprocessing"})
    else:
        log_metric("ErrorCount", 1, dimensions={"Function": "preprocessing", "ErrorType": error_type})
```

### 6.2 Generation Function Metrics

```python
# In generation.py
def generate_with_gemini(generation_type, speech_type, custom_prompt, characters):
    start_time = time.time()

    try:
        # Call Gemini API
        result = gemini_client.generate_content(...)

        # Log success metrics
        log_metric("GeminiSuccess", 1, dimensions={"GenerationType": generation_type})
        log_execution_time(start_time, "gemini_generation")

        return result
    except Exception as e:
        # Log failure and trigger fallback
        log_metric("GeminiFailure", 1, dimensions={"ErrorType": type(e).__name__})
        log_error(e, {"generation_type": generation_type})

        # Fallback to GPT-2
        return generate_with_gpt2(...)
```

### 6.3 Postprocessing Function Metrics

```python
# In postprocessing.py
def process_generation_result(job_id, result):
    start_time = time.time()

    # Store in S3
    # Update DynamoDB
    # Send notifications

    # Log storage metrics
    log_metric("StorageSize", len(result), unit="Bytes", dimensions={"Bucket": "outputs"})
    log_execution_time(start_time, "postprocessing")

    # Log notification metrics
    log_metric("NotificationSent", 1, dimensions={"Channel": "sns"})
```

## 7. CDK Configuration

### 7.1 CloudWatch Dashboard Construct

```python
# In CDK infrastructure
from aws_cdk import (
    aws_cloudwatch as cloudwatch,
    aws_cloudwatch_actions as cloudwatch_actions,
)

class MonitoringStack(cdk.Stack):
    def __init__(self, scope, id, **kwargs):
        super().__init__(scope, id, **kwargs)

        # Create dashboard
        dashboard = cloudwatch.Dashboard(
            self, "AnimeQuoteDashboard",
            dashboard_name="AnimeQuoteGenerator-Operations"
        )

        # Add widgets
        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Request Rate",
                left=[preprocessing_function.metric_invocations()],
                right=[generation_function.metric_invocations()]
            ),
            cloudwatch.GraphWidget(
                title="Error Rate",
                left=[preprocessing_function.metric_errors()],
                right=[generation_function.metric_errors()]
            )
        )
```

### 7.2 Alarm Constructs

```python
# Create alarms for each function
error_alarm = cloudwatch.Alarm(
    self, "HighErrorRate",
    metric=preprocessing_function.metric_errors(),
    threshold=10,
    evaluation_periods=2,
    datapoints_to_alarm=2,
    comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD
)

# Add actions
error_alarm.add_alarm_action(
    cloudwatch_actions.SnsAction(alarm_topic)
)
```

## 8. Cost Optimization

### 8.1 Log Retention Strategy

- Development: 7 days retention
- Staging: 14 days retention
- Production: 30 days retention
- Audit logs: 90 days retention

### 8.2 Metric Granularity

- High-frequency metrics (errors, latency): 1-minute granularity
- Business metrics (usage trends): 5-minute granularity
- Cost metrics (billing): 1-hour granularity

### 8.3 Alarm Optimization

- Critical alarms: 1-minute evaluation, immediate notification
- Warning alarms: 5-minute evaluation, batched notifications
- Informational alarms: 15-minute evaluation, daily digest

## 9. Security Considerations

### 9.1 Log Encryption

- All CloudWatch Log Groups use AWS-managed KMS encryption
- Log data encrypted at rest and in transit
- Access controlled via IAM policies with least privilege

### 9.2 Access Control

- Read-only access for developers via IAM roles
- Write access limited to Lambda execution roles
- Dashboard access restricted to operations team

### 9.3 Compliance

- Logs include request IDs for audit trails
- Retention periods comply with data governance policies
- Metric data anonymized for privacy compliance

## 10. Troubleshooting Guide

### 10.1 Common Issues

#### 10.1.1 Missing Metrics

**Symptoms**: Metrics not appearing in CloudWatch
**Resolution**:

1. Check IAM permissions for `cloudwatch:PutMetricData`
2. Verify metric namespace matches `AnimeQuoteGenerator`
3. Check Lambda execution role has CloudWatch permissions

#### 10.1.2 Alarm Not Triggering

**Symptoms**: Alarms not firing despite threshold breaches
**Resolution**:

1. Verify alarm configuration (period, evaluation periods)
2. Check metric has sufficient data points
3. Verify SNS topic permissions and subscriptions

#### 10.1.3 High Log Costs

**Symptoms**: Unexpected CloudWatch costs
**Resolution**:

1. Review log retention policies
2. Check for debug logging in production
3. Implement log level filtering
4. Consider moving historical logs to S3 for archival

### 10.2 Performance Optimization

#### 10.2.1 Reduce Metric Cardinality

- Use consistent dimension values
- Avoid high-cardinality dimensions (user IDs in metrics)
- Aggregate metrics where possible

#### 10.2.2 Optimize Log Volume

- Use structured logging instead of verbose text
- Implement log sampling for high-volume operations
- Filter out debug logs in production

## 11. Future Enhancements

### 11.1 Advanced Monitoring

- Real-time anomaly detection using CloudWatch Anomaly Detection
- Predictive scaling based on usage patterns
- Automated root cause analysis using CloudWatch Logs Insights

### 11.2 Integration with External Tools

- Export metrics to Prometheus for Grafana dashboards
- Integrate with Datadog or New Relic for APM
- Connect to PagerDuty for on-call management

### 11.3 Machine Learning Insights

- Use CloudWatch ML to predict capacity needs
- Implement automated performance optimization
- Create intelligent alerting based on historical patterns

---

## Summary

This CloudWatch configuration provides comprehensive monitoring for the Anime Quote Generator pipeline, covering logging, metrics, alarms, and dashboards. The implementation follows AWS best practices for serverless monitoring and includes cost optimization, security considerations, and troubleshooting guidance.

The monitoring system enables:

- Real-time visibility into system health and performance
- Proactive alerting for issues before they impact users
- Detailed analytics for business and technical stakeholders
- Cost-effective logging and metric collection
- Compliance with security and governance requirements
