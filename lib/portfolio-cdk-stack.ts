import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as apigateway from 'aws-cdk-lib/aws-apigateway';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as python from '@aws-cdk/aws-lambda-python-alpha';
import { bedrock } from '@cdklabs/generative-ai-cdk-constructs';
import { Construct } from 'constructs';

export class PortfolioCdkStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // ============================================
    // S3 Bucket for Resume
    // ============================================
    const resumeBucket = new s3.Bucket(this, 'ResumeBucket', {
      bucketName: `portfolio-resume-${this.account}-${this.region}`,
      removalPolicy: cdk.RemovalPolicy.DESTROY, 
      autoDeleteObjects: true, 
    });

    const sessionTable = new dynamodb.Table(this, 'SessionTable', {
      partitionKey: { name: 'sessionId', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // ============================================
    // Bedrock Knowledge Base 
    // ============================================
    const kb = new bedrock.VectorKnowledgeBase(this, 'PortfolioKBv2', {
      name: 'vz-portfolio-kb',
      embeddingsModel: bedrock.BedrockFoundationModel.TITAN_EMBED_TEXT_V2_1024,
      vectorType: bedrock.VectorType.BINARY
    });

    const dataSource = new bedrock.S3DataSource(this, 'PortfolioDataSourcev2', {
      bucket: resumeBucket,
      knowledgeBase: kb,
      dataSourceName: 'vz-resume-s3-source',
      chunkingStrategy: bedrock.ChunkingStrategy.fixedSize({
        maxTokens: 500,
        overlapPercentage: 20,
      })
    });

    // ============================================
    // Lambda Function
    // ============================================
    const chatHandler = new python.PythonFunction(this, 'ChatHandler', {
      entry: 'lambda',                    
      index: 'index.py',                  
      handler: 'lambda_handler',          
      runtime: lambda.Runtime.PYTHON_3_12,
      timeout: cdk.Duration.seconds(60),
      memorySize: 1024,                 
      environment: {
        TABLE_NAME: sessionTable.tableName,
        KNOWLEDGE_BASE_ID: kb.knowledgeBaseId,
        LOGFIRE_TOKEN: process.env.LOGFIRE_TOKEN ?? ''
      },
      bundling: {
        assetExcludes: ['boto3', 'botocore', '*.pyc', '__pycache__'],
      }
    });

    resumeBucket.grantRead(chatHandler);
    sessionTable.grantReadWriteData(chatHandler);

    chatHandler.addToRolePolicy(new iam.PolicyStatement({
      actions: [
        'bedrock:InvokeModel', 
        'bedrock:InvokeModelWithResponseStream',
        'bedrock:Retrieve'
      ],
      resources: [
        'arn:aws:bedrock:*::foundation-model/*anthropic.claude-sonnet-4-5-20250929-v1:0',
        'arn:aws:bedrock:*:*:inference-profile/*anthropic.claude-sonnet-4-5-20250929-v1:0',
        `arn:aws:bedrock:${this.region}:${this.account}:knowledge-base/*`
      ],
    }));

    // ============================================
    // API Gateway
    // ============================================
    const api = new apigateway.RestApi(this, 'PortfolioApi', {
      restApiName: 'Portfolio Assistant API',
      description: 'API for AI portfolio assistant',
      defaultCorsPreflightOptions: {
        allowOrigins: apigateway.Cors.ALL_ORIGINS,
        allowMethods: apigateway.Cors.ALL_METHODS,
        allowHeaders: ['Content-Type'],
      },
    });

    const chat = api.root.addResource('chat');
    chat.addMethod('POST', new apigateway.LambdaIntegration(chatHandler, {
      timeout: cdk.Duration.seconds(29),
    }));

    // ============================================
    // Outputs
    // ============================================
    new cdk.CfnOutput(this, 'ApiUrl', {
      value: api.url + 'chat',
      description: 'API Gateway endpoint URL',
    });

    new cdk.CfnOutput(this, 'BucketName', {
      value: resumeBucket.bucketName,
      description: 'S3 bucket for resume',
    });
  }
}