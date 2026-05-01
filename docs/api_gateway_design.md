# API Gateway HTTP API Design

## Overview

This document defines the API Gateway HTTP API configuration for the Anime Quote Generator pipeline. The API uses HTTP API (v2) for lower cost and lower latency compared to REST API, with JWT authorizer for secure access.

## 1. API Configuration

### 1.1 API Details

- **API Name**: `anime-quote-generator-api`
- **API Type**: HTTP API (v2)
- **Protocol**: HTTPS only
- **Base URL**: `https://{api-id}.execute-api.{region}.amazonaws.com/{stage}`
- **Stages**: `dev`, `staging`, `prod`

### 1.2 Custom Domain (Production)

- **Domain**: `api.animequotegenerator.com`
- **Hosted Zone**: Route 53
- **TLS Certificate**: ACM certificate in us-east-1
- **Route 53 Alias**: A-record pointing to API Gateway domain

## 2. Route Definitions

### 2.1 Generation Endpoints

| Method | Route                          | Handler       | Auth | Description                        |
| ------ | ------------------------------ | ------------- | ---- | ---------------------------------- |
| POST   | `/v1/generate`                 | preprocessing | JWT  | Generate single anime quote/speech |
| POST   | `/v1/generate/batch`           | preprocessing | JWT  | Generate batch of quotes/speeches  |
| GET    | `/v1/generate/{job_id}/status` | preprocessing | JWT  | Check generation job status        |

### 2.2 Content Endpoints

| Method | Route                   | Handler       | Auth    | Description               |
| ------ | ----------------------- | ------------- | ------- | ------------------------- |
| GET    | `/v1/quotes`            | preprocessing | API Key | List generated quotes     |
| GET    | `/v1/quotes/{quote_id}` | preprocessing | API Key | Get specific quote        |
| GET    | `/v1/quotes/random`     | preprocessing | None    | Get random quote (public) |

### 2.3 Management Endpoints

| Method | Route        | Handler       | Auth        | Description                |
| ------ | ------------ | ------------- | ----------- | -------------------------- |
| GET    | `/v1/health` | preprocessing | None        | Health check endpoint      |
| GET    | `/v1/stats`  | preprocessing | JWT         | Pipeline statistics        |
| GET    | `/v1/config` | preprocessing | JWT (Admin) | Get pipeline configuration |

### 2.4 WebSocket Endpoints (Future)

| Route         | Handler    | Description              |
| ------------- | ---------- | ------------------------ |
| `$connect`    | ws_handler | WebSocket connection     |
| `$disconnect` | ws_handler | WebSocket disconnection  |
| `subscribe`   | ws_handler | Subscribe to job updates |

## 3. Request/Response Schemas

### 3.1 POST /v1/generate

**Request Body**:

```json
{
  "generation_type": "speech",
  "speech_type": "motivational",
  "characters": ["Naruto", "Sasuke"],
  "custom_prompt": "A speech about never giving up",
  "temperature": 0.8,
  "max_length": 200,
  "callback_url": "https://example.com/webhook"
}
```

**Response (202 Accepted)**:

```json
{
  "job_id": "job-20240115-103000-abc123",
  "status": "queued",
  "generation_type": "speech",
  "speech_type": "motivational",
  "estimated_time_seconds": 15,
  "status_url": "/v1/generate/job-20240115-103000-abc123/status",
  "created_at": "2024-01-15T10:30:00Z"
}
```

### 3.2 POST /v1/generate/batch

**Request Body**:

```json
{
  "requests": [
    {
      "generation_type": "speech",
      "speech_type": "motivational",
      "characters": ["Naruto"]
    },
    {
      "generation_type": "dialogue",
      "speech_type": "villain",
      "characters": ["Hero", "Villain"]
    }
  ],
  "callback_url": "https://example.com/webhook"
}
```

**Response (202 Accepted)**:

```json
{
  "batch_id": "batch-20240115-103000-def456",
  "status": "queued",
  "total_requests": 2,
  "jobs": [
    {
      "job_id": "job-20240115-103000-abc123",
      "status": "queued",
      "status_url": "/v1/generate/job-20240115-103000-abc123/status"
    },
    {
      "job_id": "job-20240115-103000-abc124",
      "status": "queued",
      "status_url": "/v1/generate/job-20240115-103000-abc124/status"
    }
  ],
  "created_at": "2024-01-15T10:30:00Z"
}
```

### 3.3 GET /v1/generate/{job_id}/status

**Response (200 OK)**:

```json
{
  "job_id": "job-20240115-103000-abc123",
  "status": "completed",
  "generation_type": "speech",
  "speech_type": "motivational",
  "result": {
    "content": "I will never give up! That's my ninja way!",
    "character": "Naruto",
    "model_used": "gemini",
    "token_count": 42
  },
  "timing": {
    "queued_at": "2024-01-15T10:30:00Z",
    "processing_started_at": "2024-01-15T10:30:01Z",
    "completed_at": "2024-01-15T10:30:12Z",
    "total_duration_seconds": 12
  },
  "metadata": {
    "s3_key": "outputs/2024/01/15/job-20240115-103000-abc123.json",
    "request_id": "req-xyz789"
  }
}
```

### 3.4 GET /v1/quotes/random

**Response (200 OK)**:

```json
{
  "quote": {
    "content": "Believe in the me that believes in you!",
    "character": "Kamina",
    "anime": "Gurren Lagann",
    "speech_type": "motivational"
  },
  "metadata": {
    "generated_at": "2024-01-15T08:00:00Z",
    "model_used": "gemini"
  }
}
```

### 3.5 GET /v1/health

**Response (200 OK)**:

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "services": {
    "preprocessing": "healthy",
    "generation": "healthy",
    "postprocessing": "healthy",
    "s3": "healthy",
    "sqs": "healthy",
    "dynamodb": "healthy"
  }
}
```

## 4. Authentication & Authorization

### 4.1 JWT Authorizer

- **Type**: HTTP API JWT Authorizer
- **Identity Source**: Authorization header
- **Issuer**: Cognito User Pool or external OIDC provider
- **Audience**: API client ID

**Configuration**:

```json
{
  "authorizerType": "JWT",
  "identitySource": "$request.header.Authorization",
  "jwtConfiguration": {
    "issuer": "https://cognito-idp.{region}.amazonaws.com/{user-pool-id}",
    "audience": ["{client-id}"]
  }
}
```

### 4.2 API Key Authorizer

For read-only public endpoints, a simple API key approach:

- **Header**: `X-API-Key`
- **Validation**: Lambda authorizer checks key against DynamoDB
- **Rate Limiting**: Per-key rate limits

### 4.3 Route-Level Auth Mapping

```python
# Route auth configuration
ROUTE_AUTH = {
    # Public endpoints - no auth
    "GET /v1/health": None,
    "GET /v1/quotes/random": None,

    # API Key endpoints
    "GET /v1/quotes": "api_key",
    "GET /v1/quotes/{quote_id}": "api_key",

    # JWT endpoints
    "POST /v1/generate": "jwt",
    "POST /v1/generate/batch": "jwt",
    "GET /v1/generate/{job_id}/status": "jwt",
    "GET /v1/stats": "jwt",

    # Admin endpoints
    "GET /v1/config": "jwt_admin",
}
```

## 5. Rate Limiting & Throttling

### 5.1 Throttling Configuration

| Tier       | Rate (requests/second) | Burst | Monthly Quota |
| ---------- | ---------------------- | ----- | ------------- |
| Free       | 2                      | 5     | 1,000         |
| Basic      | 10                     | 20    | 10,000        |
| Premium    | 50                     | 100   | 100,000       |
| Enterprise | 200                    | 500   | Unlimited     |

### 5.2 Implementation

Rate limiting is implemented via:

1. **API Gateway**: Account-level throttling (default 10,000 rps)
2. **Lambda Authorizer**: Per-API-key rate limiting with DynamoDB counters
3. **SQS**: Queue-based backpressure for generation requests

### 5.3 Rate Limit Headers

All responses include rate limit headers:

```
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 7
X-RateLimit-Reset: 1705312200
```

### 5.4 429 Response

```json
{
  "error": "Rate limit exceeded",
  "retry_after_seconds": 30,
  "limit": 10,
  "remaining": 0,
  "reset_at": "2024-01-15T10:35:00Z"
}
```

## 6. CORS Configuration

### 6.1 CORS Settings

```json
{
  "allowOrigins": [
    "https://animequotegenerator.com",
    "https://app.animequotegenerator.com"
  ],
  "allowMethods": ["GET", "POST", "OPTIONS"],
  "allowHeaders": ["Content-Type", "Authorization", "X-API-Key"],
  "exposeHeaders": [
    "X-RateLimit-Limit",
    "X-RateLimit-Remaining",
    "X-RateLimit-Reset"
  ],
  "maxAge": 86400,
  "allowCredentials": true
}
```

### 6.2 Development CORS

For development, allow all origins:

```json
{
  "allowOrigins": ["*"],
  "allowMethods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
  "allowHeaders": ["*"],
  "maxAge": 3600
}
```

## 7. Request Validation

### 7.1 Input Validation Rules

```json
{
  "POST /v1/generate": {
    "generation_type": {
      "required": true,
      "type": "string",
      "enum": ["speech", "dialogue", "quote", "narration"]
    },
    "speech_type": {
      "required": false,
      "type": "string",
      "enum": [
        "motivational",
        "villain",
        "heroic",
        "comedic",
        "dramatic",
        "philosophical"
      ]
    },
    "characters": {
      "required": false,
      "type": "array",
      "maxItems": 5,
      "items": {
        "type": "string",
        "maxLength": 50
      }
    },
    "custom_prompt": {
      "required": false,
      "type": "string",
      "maxLength": 1000
    },
    "temperature": {
      "required": false,
      "type": "number",
      "minimum": 0.1,
      "maximum": 2.0,
      "default": 0.8
    },
    "max_length": {
      "required": false,
      "type": "integer",
      "minimum": 50,
      "maximum": 1000,
      "default": 200
    }
  }
}
```

### 7.2 Validation Error Response

```json
{
  "error": "Validation failed",
  "details": [
    {
      "field": "generation_type",
      "message": "Must be one of: speech, dialogue, quote, narration",
      "received": "invalid_type"
    },
    {
      "field": "temperature",
      "message": "Must be between 0.1 and 2.0",
      "received": 3.5
    }
  ]
}
```

## 8. Error Responses

### 8.1 Standard Error Format

```json
{
  "error": "Error description",
  "code": "ERROR_CODE",
  "details": {},
  "request_id": "req-abc123",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### 8.2 Error Codes

| HTTP Status | Code                  | Description                        |
| ----------- | --------------------- | ---------------------------------- |
| 400         | `VALIDATION_ERROR`    | Invalid request body or parameters |
| 401         | `UNAUTHORIZED`        | Missing or invalid authentication  |
| 403         | `FORBIDDEN`           | Insufficient permissions           |
| 404         | `NOT_FOUND`           | Resource not found                 |
| 429         | `RATE_LIMITED`        | Too many requests                  |
| 500         | `INTERNAL_ERROR`      | Server-side processing error       |
| 502         | `UPSTREAM_ERROR`      | External service failure           |
| 503         | `SERVICE_UNAVAILABLE` | Service temporarily unavailable    |
| 504         | `TIMEOUT`             | Request processing timeout         |

## 9. Integration Mapping

### 9.1 Request Mapping

API Gateway forwards requests to Lambda with the following structure:

```json
{
  "version": "2.0",
  "routeKey": "POST /v1/generate",
  "rawPath": "/v1/generate",
  "rawQueryString": "",
  "headers": {
    "content-type": "application/json",
    "authorization": "Bearer ..."
  },
  "requestContext": {
    "http": {
      "method": "POST",
      "path": "/v1/generate"
    },
    "authorizer": {
      "jwt": {
        "claims": {
          "sub": "user-id",
          "email": "user@example.com"
        }
      }
    },
    "requestId": "req-abc123",
    "stage": "prod"
  },
  "body": "{\"generation_type\":\"speech\",\"speech_type\":\"motivational\"}",
  "isBase64Encoded": false
}
```

### 9.2 Response Mapping

Lambda returns responses in API Gateway v2 format:

```json
{
  "statusCode": 202,
  "headers": {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*"
  },
  "body": "{\"job_id\":\"job-123\",\"status\":\"queued\"}",
  "isBase64Encoded": false
}
```

## 10. CDK Implementation

### 10.1 HTTP API Construct

```python
from aws_cdk import (
    aws_apigatewayv2 as apigwv2,
    aws_apigatewayv2_integrations as integrations,
)

class ApiGatewayConstruct(cdk.Construct):
    def __init__(self, scope, id, **kwargs):
        super().__init__(scope, id, **kwargs)

        # Create HTTP API
        self.api = apigwv2.HttpApi(
            self, "AnimeQuoteApi",
            api_name="anime-quote-generator-api",
            description="Anime Quote Generator Processing Pipeline API",
            cors_preflight=apigwv2.CorsPreflightOptions(
                allow_origins=["*"],
                allow_methods=[apigwv2.CorsHttpMethod.ANY],
                allow_headers=["*"],
                max_age=cdk.Duration.hours(1)
            )
        )

        # Add routes
        self._add_routes()

    def _add_routes(self):
        # Generation endpoints
        self.api.add_routes(
            path="/v1/generate",
            methods=[apigwv2.HttpMethod.POST],
            integration=integrations.HttpLambdaIntegration(
                "PreprocessingIntegration",
                self.preprocessing_handler
            ),
            authorizer=self.jwt_authorizer
        )

        self.api.add_routes(
            path="/v1/generate/batch",
            methods=[apigwv2.HttpMethod.POST],
            integration=integrations.HttpLambdaIntegration(
                "BatchPreprocessingIntegration",
                self.preprocessing_handler
            ),
            authorizer=self.jwt_authorizer
        )

        self.api.add_routes(
            path="/v1/generate/{job_id}/status",
            methods=[apigwv2.HttpMethod.GET],
            integration=integrations.HttpLambdaIntegration(
                "StatusIntegration",
                self.preprocessing_handler
            ),
            authorizer=self.jwt_authorizer
        )

        # Public endpoints
        self.api.add_routes(
            path="/v1/health",
            methods=[apigwv2.HttpMethod.GET],
            integration=integrations.HttpLambdaIntegration(
                "HealthIntegration",
                self.preprocessing_handler
            )
        )

        self.api.add_routes(
            path="/v1/quotes/random",
            methods=[apigwv2.HttpMethod.GET],
            integration=integrations.HttpLambdaIntegration(
                "RandomQuoteIntegration",
                self.preprocessing_handler
            )
        )
```

## 11. Monitoring & Logging

### 11.1 Access Logging

```json
{
  "accessLogSettings": {
    "destinationArn": "arn:aws:logs:{region}:{account}:log-group:/aws/apigateway/anime-quote-api",
    "format": "{\"requestId\":\"$context.requestId\",\"ip\":\"$context.identity.sourceIp\",\"requestTime\":\"$context.requestTime\",\"httpMethod\":\"$context.httpMethod\",\"routeKey\":\"$context.routeKey\",\"status\":\"$context.status\",\"responseLength\":\"$context.responseLength\",\"integrationErrorMessage\":\"$context.integrationErrorMessage\"}"
  }
}
```

### 11.2 Key Metrics

| Metric                 | Alarm          | Threshold       |
| ---------------------- | -------------- | --------------- |
| 4xxErrorRate           | High 4xx rate  | >5% over 5 min  |
| 5xxErrorRate           | High 5xx rate  | >1% over 5 min  |
| Latency p99            | Slow responses | >10s over 5 min |
| IntegrationLatency p99 | Slow Lambda    | >8s over 5 min  |
| Count                  | High traffic   | >1000 rps       |

---

## Summary

This API Gateway design provides a secure, scalable, and well-documented HTTP API for the Anime Quote Generator pipeline. Key features include:

- RESTful route design with versioned endpoints
- JWT and API Key authentication
- Per-tier rate limiting and throttling
- Comprehensive CORS configuration
- Input validation with clear error messages
- Async processing pattern with job status polling
- Full CDK implementation for infrastructure-as-code
