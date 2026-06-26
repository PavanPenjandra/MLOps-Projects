"""
AWS deployment script using boto3 and CDK patterns.
"""

import boto3
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AWSDeployment:
    """Handle AWS deployment operations."""

    def __init__(self, region: str = "us-east-1", aws_profile: str = None):
        """
        Initialize AWS deployment.

        Args:
            region: AWS region
            aws_profile: AWS profile name
        """
        self.region = region
        session = boto3.Session(profile_name=aws_profile, region_name=region)

        self.s3_client = session.client("s3")
        self.ecr_client = session.client("ecr")
        self.sagemaker_client = session.client("sagemaker")
        self.cloudformation_client = session.client("cloudformation")

    def push_image_to_ecr(
        self, image_uri: str, repository_name: str = "nlp-sentiment-analysis"
    ) -> str:
        """
        Push Docker image to ECR.

        Args:
            image_uri: Local Docker image URI
            repository_name: ECR repository name

        Returns:
            ECR image URI
        """
        logger.info(f"Pushing image {image_uri} to ECR")

        # Get ECR login
        import subprocess

        # Get authorization token
        auth_data = self.ecr_client.get_authorization_token()
        auth_token = auth_data["authorizationData"][0]["authorizationToken"]

        # Login to ECR
        registry = auth_data["authorizationData"][0]["proxyEndpoint"]
        username, password = "AWS", auth_token

        login_cmd = (
            f"aws ecr get-login-password --region {self.region} | "
            f"docker login --username AWS --password-stdin {registry}"
        )
        subprocess.run(login_cmd, shell=True, check=True)

        # Push image
        account_id = auth_data["authorizationData"][0]["proxyEndpoint"].split(".")[0]
        ecr_uri = (
            f"{account_id}.dkr.ecr.{self.region}.amazonaws.com/{repository_name}:latest"
        )

        tag_cmd = f"docker tag {image_uri} {ecr_uri}"
        push_cmd = f"docker push {ecr_uri}"

        subprocess.run(tag_cmd, shell=True, check=True)
        subprocess.run(push_cmd, shell=True, check=True)

        logger.info(f"Successfully pushed to {ecr_uri}")
        return ecr_uri

    def upload_model_to_s3(
        self,
        model_dir: str,
        bucket_name: str = "nlp-sentiment-models",
        s3_prefix: str = "models",
    ) -> str:
        """
        Upload model artifacts to S3.

        Args:
            model_dir: Local model directory
            bucket_name: S3 bucket name
            s3_prefix: S3 prefix path

        Returns:
            S3 model URI
        """
        logger.info(f"Uploading model from {model_dir} to S3")

        model_path = Path(model_dir)

        # Create bucket if not exists
        try:
            self.s3_client.create_bucket(Bucket=bucket_name)
            logger.info(f"Created S3 bucket {bucket_name}")
        except self.s3_client.exceptions.BucketAlreadyOwnedByYou:
            logger.info(f"S3 bucket {bucket_name} already exists")

        # Upload files
        for file_path in model_path.rglob("*"):
            if file_path.is_file():
                s3_key = f"{s3_prefix}/{file_path.relative_to(model_path)}"
                self.s3_client.upload_file(str(file_path), bucket_name, s3_key)
                logger.info(f"Uploaded {s3_key}")

        s3_uri = f"s3://{bucket_name}/{s3_prefix}"
        logger.info(f"Model uploaded to {s3_uri}")
        return s3_uri

    def create_sagemaker_model(
        self,
        model_name: str,
        image_uri: str,
        model_s3_uri: str,
        execution_role_arn: str,
    ) -> str:
        """
        Create SageMaker model.

        Args:
            model_name: Model name
            image_uri: ECR image URI
            model_s3_uri: S3 model URI
            execution_role_arn: IAM role ARN

        Returns:
            Model ARN
        """
        logger.info(f"Creating SageMaker model {model_name}")

        response = self.sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                "Image": image_uri,
                "ModelDataUrl": model_s3_uri,
            },
            ExecutionRoleArn=execution_role_arn,
        )

        logger.info(f"Created model: {response['ModelArn']}")
        return response["ModelArn"]

    def deploy_stack(
        self, stack_name: str, template_path: str, parameters: dict
    ) -> str:
        """
        Deploy CloudFormation stack.

        Args:
            stack_name: Stack name
            template_path: Path to CloudFormation template
            parameters: Stack parameters

        Returns:
            Stack ID
        """
        logger.info(f"Deploying CloudFormation stack {stack_name}")

        with open(template_path, "r") as f:
            template_body = f.read()

        cf_params = [
            {"ParameterKey": k, "ParameterValue": v} for k, v in parameters.items()
        ]

        response = self.cloudformation_client.create_stack(
            StackName=stack_name,
            TemplateBody=template_body,
            Parameters=cf_params,
            Capabilities=["CAPABILITY_IAM"],
        )

        logger.info(f"Stack created: {response['StackId']}")
        return response["StackId"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--image-uri", type=str, required=True)
    args = parser.parse_args()

    deployer = AWSDeployment(region=args.region)

    # Upload model to S3
    s3_uri = deployer.upload_model_to_s3(args.model_dir)
    logger.info(f"Model S3 URI: {s3_uri}")

    # Push image to ECR
    ecr_uri = deployer.push_image_to_ecr(args.image_uri)
    logger.info(f"ECR URI: {ecr_uri}")
