#!/bin/bash
# Deployment script for AWS

set -e

# Configuration
REGION=${AWS_REGION:-us-east-1}
MODEL_DIR=${MODEL_DIR:-models/sentiment-model}
IMAGE_NAME=${IMAGE_NAME:-nlp-sentiment-analysis}
STACK_NAME=${STACK_NAME:-sentiment-analysis-stack}

echo "NLP Sentiment Analysis - AWS Deployment"
echo "========================================"
echo "Region: $REGION"
echo "Model Directory: $MODEL_DIR"
echo "Image Name: $IMAGE_NAME"
echo ""

# Step 1: Build Docker image
echo "Step 1: Building Docker image..."
docker build -t $IMAGE_NAME:latest .

# Step 2: Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPOSITORY=$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$IMAGE_NAME

# Step 3: Create ECR repository if it doesn't exist
echo "Step 2: Ensuring ECR repository exists..."
aws ecr describe-repositories --repository-names $IMAGE_NAME --region $REGION 2>/dev/null || \
  aws ecr create-repository --repository-name $IMAGE_NAME --region $REGION

# Step 4: Login to ECR
echo "Step 3: Logging in to ECR..."
aws ecr get-login-password --region $REGION | \
  docker login --username AWS --password-stdin $ECR_REPOSITORY

# Step 5: Tag and push image
echo "Step 4: Pushing Docker image to ECR..."
docker tag $IMAGE_NAME:latest $ECR_REPOSITORY:latest
docker tag $IMAGE_NAME:latest $ECR_REPOSITORY:$(date +%Y%m%d_%H%M%S)
docker push $ECR_REPOSITORY:latest

# Step 6: Upload model to S3
echo "Step 5: Uploading model to S3..."
S3_BUCKET=nlp-sentiment-models-$ACCOUNT_ID
aws s3 mb s3://$S3_BUCKET --region $REGION 2>/dev/null || true
aws s3 cp $MODEL_DIR s3://$S3_BUCKET/models/ --recursive

# Step 7: Deploy with CloudFormation
echo "Step 6: Deploying with CloudFormation..."
aws cloudformation deploy \
  --template-file aws/cloudformation_template.json \
  --stack-name $STACK_NAME \
  --parameter-overrides \
    ECRImageUri=$ECR_REPOSITORY:latest \
    SageMakerExecutionRole=<your-sagemaker-role-arn> \
    InstanceType=ml.m5.large \
  --region $REGION \
  --capabilities CAPABILITY_IAM

echo ""
echo "Deployment complete!"
echo "SageMaker Endpoint: $(aws cloudformation describe-stacks --stack-name $STACK_NAME --region $REGION --query 'Stacks[0].Outputs[?OutputKey==`SageMakerEndpoint`].OutputValue' --output text)"
