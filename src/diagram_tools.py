"""
Diagram Tools Module

This module provides tools for generating infrastructure diagrams based on natural language descriptions.
It supports both AWS and GCP cloud providers, with their respective icons and components.

Key Features:
1. Provider-specific icon mapping
2. Automatic diagram generation
3. Support for clustering and connections
4. Validation of provider-specific components

The module uses the 'diagrams' library to create visual representations of cloud infrastructure.
"""

from typing import List, Dict, Optional
from diagrams import Diagram, Cluster, Edge
from diagrams.aws.compute import EC2 as AWS_EC2, Lambda as AWS_Lambda, ECS as AWS_ECS, Fargate as AWS_Fargate, Batch as AWS_Batch, AppRunner as AWS_AppRunner, ElasticBeanstalk as AWS_ElasticBeanstalk, EKS as AWS_EKS, ECR as AWS_ECR, ComputeOptimizer as AWS_ComputeOptimizer, Compute as AWS_Compute
from diagrams.aws.database import RDS as AWS_RDS, Dynamodb as AWS_Dynamodb, RedshiftDenseComputeNode as AWS_RedshiftDenseComputeNode, RedshiftDenseStorageNode as AWS_RedshiftDenseStorageNode, QLDB as AWS_QLDB, Aurora as AWS_Aurora, ElastiCache as AWS_ElastiCache, Neptune as AWS_Neptune
from diagrams.aws.network import (
    VPC as AWS_VPC, ELB as AWS_ELB, APIGateway as AWS_APIGateway,
    Route53 as AWS_Route53, CloudFront as AWS_CloudFront,
    DirectConnect as AWS_DirectConnect, TransitGateway as AWS_TransitGateway,
    NATGateway as AWS_NATGateway, DirectConnect as AWS_NetDirectConnect,
    TransitGateway as AWS_NetTransitGateway, NATGateway as AWS_NetNATGateway
)
from diagrams.aws.storage import S3 as AWS_S3, EFS as AWS_EFS, FSx as AWS_FSx, StorageGateway as AWS_StorageGateway, Backup as AWS_Backup, Snowball as AWS_Snowball, Snowmobile as AWS_Snowmobile, Storage as AWS_Storage, Backup as AWS_StorageBackup, EFS as AWS_StorageEFS, FSx as AWS_StorageFSx, S3 as AWS_StorageS3, Snowball as AWS_StorageSnowball, Snowmobile as AWS_StorageSnowmobile, Storage as AWS_StorageStorage
from diagrams.aws.security import CertificateManager as AWS_CertificateManager, Inspector as AWS_Inspector, Macie as AWS_Macie, SecurityHub as AWS_SecurityHub, Shield as AWS_Shield, WAF as AWS_WAF, KMS as AWS_KMS, SecretsManager as AWS_SecretsManager, IAM as AWS_IAM, IAMRole as AWS_IAMRole
from diagrams.aws.integration import SQS as AWS_SQS, SNS as AWS_SNS, StepFunctions as AWS_StepFunctions, MQ as AWS_MQ
from diagrams.aws.analytics import (
    Athena as AWS_Athena, Glue as AWS_Glue, Quicksight as AWS_Quicksight,
    EMR as AWS_EMR, DataPipeline as AWS_DataPipeline, DataLakeResource as AWS_DataLakeResource,
    Cloudsearch as AWS_Cloudsearch, ElasticsearchService as AWS_ElasticsearchService,
    Redshift as AWS_Redshift, Kinesis as AWS_Kinesis
)
from diagrams.aws.management import (
    Cloudwatch as AWS_Cloudwatch, Cloudtrail as AWS_Cloudtrail,
    SystemsManager as AWS_SystemsManager, Config as AWS_Config,
    ControlTower as AWS_ControlTower, LicenseManager as AWS_LicenseManager,
    ManagedServices as AWS_ManagedServices, Organizations as AWS_Organizations,
    ServiceCatalog as AWS_ServiceCatalog, TrustedAdvisor as AWS_TrustedAdvisor,
    AutoScaling as AWS_AutoScaling, Config as AWS_MgmtConfig,
    LicenseManager as AWS_MgmtLicenseManager, ManagedServices as AWS_MgmtManagedServices,
    Organizations as AWS_MgmtOrganizations, ServiceCatalog as AWS_MgmtServiceCatalog,
    SystemsManager as AWS_MgmtSystemsManager, TrustedAdvisor as AWS_MgmtTrustedAdvisor
)
from diagrams.aws.iot import IotCore as AWS_IotCore, IotAnalytics as AWS_IotAnalytics
from diagrams.aws.ml import (
    Sagemaker as AWS_Sagemaker, Comprehend as AWS_Comprehend,
    Lex as AWS_Lex, Polly as AWS_Polly, Rekognition as AWS_Rekognition,
    Textract as AWS_Textract, Translate as AWS_Translate,
    Transcribe as AWS_Transcribe, Personalize as AWS_Personalize,
    Forecast as AWS_Forecast, Kendra as AWS_Kendra,
    MachineLearning as AWS_MachineLearning, DeepLearningContainers as AWS_DeepLearningContainers,
    Comprehend as AWS_MLComprehend, Lex as AWS_MLLex,
    Polly as AWS_MLPolly, Rekognition as AWS_MLRekognition,
    Textract as AWS_MLTextract, Translate as AWS_MLTranslate,
    Transcribe as AWS_MLTranscribe, Personalize as AWS_MLPersonalize,
    Forecast as AWS_MLForecast, Kendra as AWS_MLKendra,
    MachineLearning as AWS_MLMachineLearning, DeepLearningContainers as AWS_MLDeepLearningContainers
)
from diagrams.aws.devtools import Cloud9 as AWS_Cloud9, XRay as AWS_XRay
from diagrams.aws.business import AlexaForBusiness as AWS_AlexaForBusiness, Chime as AWS_Chime, Workmail as AWS_Workmail
from diagrams.aws.ar import Sumerian as AWS_Sumerian
from diagrams.aws.blockchain import ManagedBlockchain as AWS_ManagedBlockchain, QLDB as AWS_BlockchainQLDB
from diagrams.aws.migration import DMS as AWS_DMS, MigrationHub as AWS_MigrationHub, Snowball as AWS_MigrationSnowball
from diagrams.aws.media import ElasticTranscoder as AWS_MediaElasticTranscoder, KinesisVideoStreams as AWS_MediaKinesisVideoStreams
from diagrams.aws.mobile import Amplify as AWS_Amplify, DeviceFarm as AWS_DeviceFarm, Pinpoint as AWS_Pinpoint
from diagrams.aws.quantum import Braket as AWS_Braket
from diagrams.aws.satellite import GroundStation as AWS_GroundStation
from diagrams.aws.engagement import Connect as AWS_Connect, Pinpoint as AWS_EngagementPinpoint
from diagrams.aws.management import AutoScaling, Config as MgmtConfig, LicenseManager as MgmtLicenseManager, ManagedServices as MgmtManagedServices, Organizations as MgmtOrganizations, ServiceCatalog as MgmtServiceCatalog, SystemsManager as MgmtSystemsManager, TrustedAdvisor as MgmtTrustedAdvisor
from diagrams.aws.media import ElasticTranscoder as MediaElasticTranscoder, KinesisVideoStreams as MediaKinesisVideoStreams
from diagrams.aws.ml import Comprehend as MLComprehend, Lex as MLLex, Polly as MLPolly, Rekognition as MLRekognition, Textract as MLTextract, Translate as MLTranslate, Transcribe as MLTranscribe, Personalize as MLPersonalize, Forecast as MLForecast, Kendra as MLKendra, MachineLearning as MLMachineLearning, DeepLearningContainers as MLDeepLearningContainers
from diagrams.aws.network import DirectConnect as NetDirectConnect, TransitGateway as NetTransitGateway, NATGateway as NetNATGateway
from diagrams.aws.storage import Backup as StorageBackup, EFS as StorageEFS, FSx as StorageFSx, S3 as StorageS3, Snowball as StorageSnowball, Snowmobile as StorageSnowmobile, Storage as StorageStorage
from diagrams.gcp.analytics import (
    BigQuery as GCP_BigQuery, Composer as GCP_Composer,
    DataCatalog as GCP_DataCatalog, DataFusion as GCP_DataFusion,
    Dataflow as GCP_Dataflow, Datalab as GCP_Datalab,
    Dataprep as GCP_Dataprep, Dataproc as GCP_Dataproc,
    Genomics as GCP_Genomics, PubSub as GCP_PubSub
)
from diagrams.gcp.api import APIGateway as GCP_APIGateway, Endpoints as GCP_Endpoints
from diagrams.gcp.compute import (
    AppEngine as GCP_AppEngine, ComputeEngine as GCP_ComputeEngine,
    ContainerOptimizedOS as GCP_ContainerOptimizedOS, Functions as GCP_Functions,
    GKEOnPrem as GCP_GKEOnPrem, GPU as GCP_GPU,
    KubernetesEngine as GCP_KubernetesEngine, Run as GCP_Run,
    GKE as GCP_GKE
)
from diagrams.gcp.database import (
    Bigtable as GCP_Bigtable, Datastore as GCP_Datastore,
    Firestore as GCP_Firestore, Memorystore as GCP_Memorystore,
    Spanner as GCP_Spanner, SQL as GCP_SQL
)
from diagrams.gcp.devtools import (
    Build as GCP_Build, CodeForIntellij as GCP_CodeForIntellij,
    Code as GCP_Code, ContainerRegistry as GCP_ContainerRegistry,
    GradleAppEnginePlugin as GCP_GradleAppEnginePlugin,
    IdePlugins as GCP_IdePlugins,
    MavenAppEnginePlugin as GCP_MavenAppEnginePlugin,
    Scheduler as GCP_Scheduler, SDK as GCP_SDK,
    SourceRepositories as GCP_SourceRepositories,
    Tasks as GCP_Tasks, TestLab as GCP_TestLab,
    ToolsForEclipse as GCP_ToolsForEclipse,
    ToolsForPowershell as GCP_ToolsForPowershell,
    ToolsForVisualStudio as GCP_ToolsForVisualStudio
)
from diagrams.gcp.iot import IotCore as GCP_IotCore
from diagrams.gcp.migration import TransferAppliance as GCP_TransferAppliance
from diagrams.gcp.ml import (
    AdvancedSolutionsLab as GCP_AdvancedSolutionsLab,
    AIHub as GCP_AIHub,
    AIPlatformDataLabelingService as GCP_AIPlatformDataLabelingService,
    AIPlatform as GCP_AIPlatform,
    AutomlNaturalLanguage as GCP_AutomlNaturalLanguage,
    AutomlTables as GCP_AutomlTables,
    AutomlTranslation as GCP_AutomlTranslation,
    AutomlVideoIntelligence as GCP_AutomlVideoIntelligence,
    AutomlVision as GCP_AutomlVision,
    Automl as GCP_Automl,
    DialogFlowEnterpriseEdition as GCP_DialogFlowEnterpriseEdition,
    InferenceAPI as GCP_InferenceAPI,
    JobsAPI as GCP_JobsAPI,
    NaturalLanguageAPI as GCP_NaturalLanguageAPI,
    RecommendationsAI as GCP_RecommendationsAI,
    SpeechToText as GCP_SpeechToText,
    TextToSpeech as GCP_TextToSpeech,
    TPU as GCP_TPU,
    TranslationAPI as GCP_TranslationAPI,
    VideoIntelligenceAPI as GCP_VideoIntelligenceAPI,
    VisionAPI as GCP_VisionAPI
)
from diagrams.gcp.network import (
    Armor as GCP_Armor, CDN as GCP_CDN,
    DedicatedInterconnect as GCP_DedicatedInterconnect,
    DNS as GCP_DNS, ExternalIpAddresses as GCP_ExternalIpAddresses,
    FirewallRules as GCP_FirewallRules,
    LoadBalancing as GCP_LoadBalancing, NAT as GCP_NAT,
    Network as GCP_Network,
    PartnerInterconnect as GCP_PartnerInterconnect,
    PremiumNetworkTier as GCP_PremiumNetworkTier,
    Router as GCP_Router, Routes as GCP_Routes,
    StandardNetworkTier as GCP_StandardNetworkTier,
    TrafficDirector as GCP_TrafficDirector,
    VirtualPrivateCloud as GCP_VirtualPrivateCloud,
    VPN as GCP_VPN
)
from diagrams.gcp.operations import Monitoring as GCP_Monitoring
from diagrams.gcp.security import (
    Iam as GCP_Iam, IAP as GCP_IAP,
    KeyManagementService as GCP_KeyManagementService,
    ResourceManager as GCP_ResourceManager,
    SecurityCommandCenter as GCP_SecurityCommandCenter,
    SecurityScanner as GCP_SecurityScanner
)
from diagrams.gcp.storage import (
    Filestore as GCP_Filestore,
    PersistentDisk as GCP_PersistentDisk,
    Storage as GCP_Storage,
    GCS as GCP_GCS
)
from diagrams.onprem.compute import Server
from diagrams.onprem.database import PostgreSQL
from diagrams.onprem.network import Internet

import os
import logging
import uuid
import json

logger = logging.getLogger(__name__)

class DiagramGenerator:
    """
    A tool for generating infrastructure diagrams based on natural language descriptions.
    
    This class provides functionality to:
    1. Map natural language descriptions to cloud service icons
    2. Generate diagrams with proper clustering and connections
    3. Validate provider-specific components
    4. Support both AWS and GCP cloud providers
    
    The generator ensures that:
    - Only provider-specific icons are used
    - Components are properly clustered
    - Connections are correctly represented
    - Diagrams are visually appealing and professional
    """
    
    def __init__(self):
        """
        Initialize the diagram generator with provider-specific node mappings.
        
        The mappings include:
        - AWS services and their icons
        - GCP services and their icons
        - Common components that can be used in both
        """
        self.aws_node_mapping = {
            # AWS nodes
            'ec2': AWS_EC2,
            'lambda': AWS_Lambda,
            'rds': AWS_RDS,
            'vpc': AWS_VPC,
            's3': AWS_S3,
            'elb': AWS_ELB,  # Application Load Balancer
            'alb': AWS_ELB,  # Alias for ALB
            'apigateway': AWS_APIGateway,
            'api gateway': AWS_APIGateway,  # Alternative name
            'sqs': AWS_SQS,
            'sns': AWS_SNS,
            'message queue': AWS_SQS,  # Alternative name
            'cloudwatch': AWS_Cloudwatch,
            'monitoring': AWS_Cloudwatch,  # Alternative name
            'redshift': AWS_Redshift,
            'dynamodb': AWS_Dynamodb,
            'iam': AWS_IAM,
            'iam role': AWS_IAMRole,
            'role': AWS_IAMRole,
            'route53': AWS_Route53,
            'dns': AWS_Route53,
            'cloudfront': AWS_CloudFront,
            'cdn': AWS_CloudFront,
            'ecs': AWS_ECS,
            'fargate': AWS_Fargate,
            'cloudtrail': AWS_Cloudtrail,
            'iot core': AWS_IotCore,
            'aws iot core': AWS_IotCore,
            'kinesis': AWS_Kinesis,
            'kinesis data streams': AWS_Kinesis,
            'athena': AWS_Athena,
            'amazon athena': AWS_Athena,
            # AWS analytics
            'glue': AWS_Glue,
            'quicksight': AWS_Quicksight,
            'emr': AWS_EMR,
            'data pipeline': AWS_DataPipeline,
            'data lake resource': AWS_DataLakeResource,
            'cloud search': AWS_Cloudsearch,
            'elasticsearch service': AWS_ElasticsearchService,
            # AWS compute
            'batch': AWS_Batch,
            'app runner': AWS_AppRunner,
            'elastic beanstalk': AWS_ElasticBeanstalk,
            'eks': AWS_EKS,
            'ecr': AWS_ECR,
            'compute optimizer': AWS_ComputeOptimizer,
            'compute': AWS_Compute,
            # AWS database
            'redshift dense compute node': AWS_RedshiftDenseComputeNode,
            'redshift dense storage node': AWS_RedshiftDenseStorageNode,
            'qldb': AWS_QLDB,
            'aurora': AWS_Aurora,
            'elasti cache': AWS_ElastiCache,
            'neptune': AWS_Neptune,
            # AWS network
            'direct connect': AWS_DirectConnect,
            'transit gateway': AWS_TransitGateway,
            'nat gateway': AWS_NATGateway,
            # AWS storage
            'efs': AWS_EFS,
            'fsx': AWS_FSx,
            'storage gateway': AWS_StorageGateway,
            'backup': AWS_Backup,
            'snowball': AWS_Snowball,
            'snowmobile': AWS_Snowmobile,
            # AWS security
            'kms': AWS_KMS,
            'waf': AWS_WAF,
            'shield': AWS_Shield,
            'security hub': AWS_SecurityHub,
            'macie': AWS_Macie,
            'inspector': AWS_Inspector,
            'secrets manager': AWS_SecretsManager,
            'certificate manager': AWS_CertificateManager,
            # AWS integration
            'step functions': AWS_StepFunctions,
            'mq': AWS_MQ,
            # AWS management
            'systems manager': AWS_SystemsManager,
            'config': AWS_Config,
            'control tower': AWS_ControlTower,
            'license manager': AWS_LicenseManager,
            'managed services': AWS_ManagedServices,
            'organizations': AWS_Organizations,
            'service catalog': AWS_ServiceCatalog,
            'trusted advisor': AWS_TrustedAdvisor,
            # AWS iot
            'iot analytics': AWS_IotAnalytics,
            # AWS ml
            'sagemaker': AWS_Sagemaker,
            'comprehend': AWS_Comprehend,
            'lex': AWS_Lex,
            'polly': AWS_Polly,
            'rekognition': AWS_Rekognition,
            'textract': AWS_Textract,
            'translate': AWS_Translate,
            'transcribe': AWS_Transcribe,
            'personalize': AWS_Personalize,
            'forecast': AWS_Forecast,
            'kendra': AWS_Kendra,
            'machine learning': AWS_MachineLearning,
            'deep learning containers': AWS_DeepLearningContainers,
            # AWS devtools
            'cloud9': AWS_Cloud9,
            'xray': AWS_XRay,
            # AWS business
            'alexa for business': AWS_AlexaForBusiness,
            'chime': AWS_Chime,
            'workmail': AWS_Workmail,
            # AWS ar
            'sumerian': AWS_Sumerian,
            # AWS blockchain
            'managed blockchain': AWS_ManagedBlockchain,
            'blockchain qldb': AWS_BlockchainQLDB,
            # AWS migration
            'dms': AWS_DMS,
            'migration hub': AWS_MigrationHub,
            'migration snowball': AWS_MigrationSnowball,
            # AWS media
            'elastic transcoder': AWS_MediaElasticTranscoder,
            'kinesis video streams': AWS_MediaKinesisVideoStreams,
            # AWS mobile
            'amplify': AWS_Amplify,
            'device farm': AWS_DeviceFarm,
            'pinpoint': AWS_Pinpoint,
            # AWS quantum
            'braket': AWS_Braket,
            # AWS satellite
            'ground station': AWS_GroundStation,
            # AWS engagement
            'engagement pinpoint': AWS_EngagementPinpoint,
            # AWS management
            'auto scaling': AWS_AutoScaling,
            'mgmt config': AWS_MgmtConfig,
            'mgmt license manager': AWS_MgmtLicenseManager,
            'mgmt managed services': AWS_MgmtManagedServices,
            'mgmt organizations': AWS_MgmtOrganizations,
            'mgmt service catalog': AWS_MgmtServiceCatalog,
            'mgmt systems manager': AWS_MgmtSystemsManager,
            'mgmt trusted advisor': AWS_MgmtTrustedAdvisor,
            # AWS media
            'media elastic transcoder': AWS_MediaElasticTranscoder,
            'media kinesis video streams': AWS_MediaKinesisVideoStreams,
            # AWS ml
            'ml comprehend': AWS_MLComprehend,
            'ml lex': AWS_MLLex,
            'ml polly': AWS_MLPolly,
            'ml rekognition': AWS_MLRekognition,
            'ml textract': AWS_MLTextract,
            'ml translate': AWS_MLTranslate,
            'ml transcribe': AWS_MLTranscribe,
            'ml personalize': AWS_MLPersonalize,
            'ml forecast': AWS_MLForecast,
            'ml kendra': AWS_MLKendra,
            'ml machine learning': AWS_MLMachineLearning,
            'ml deep learning containers': AWS_MLDeepLearningContainers,
            # AWS network
            'net direct connect': AWS_NetDirectConnect,
            'net transit gateway': AWS_NetTransitGateway,
            'net nat gateway': AWS_NetNATGateway,
            # AWS security (Sec* aliases)
            'sec certificate manager': AWS_CertificateManager,
            'sec inspector': AWS_Inspector,
            'sec macie': AWS_Macie,
            'sec security hub': AWS_SecurityHub,
            'sec shield': AWS_Shield,
            'sec waf': AWS_WAF,
            # AWS storage
            'storage backup': AWS_StorageBackup,
            'storage efs': AWS_StorageEFS,
            'storage fsx': AWS_StorageFSx,
            'storage s3': AWS_StorageS3,
            'storage snowball': AWS_StorageSnowball,
            'storage snowmobile': AWS_StorageSnowmobile,
            'storage storage': AWS_StorageStorage,
        }

        self.gcp_node_mapping = {
            # GCP nodes
            'compute': GCP_ComputeEngine,
            'compute engine': GCP_ComputeEngine,
            'gce': GCP_ComputeEngine,
            'google compute engine': GCP_ComputeEngine,
            'google vm': GCP_ComputeEngine,
            'vm': GCP_ComputeEngine,
            'virtual machine': GCP_ComputeEngine,
            'functions': GCP_Functions,
            'function': GCP_Functions,
            'cloud function': GCP_Functions,
            'cloud functions': GCP_Functions,
            'gcf': GCP_Functions,
            'google cloud function': GCP_Functions,
            'google cloud functions': GCP_Functions,
            'faas': GCP_Functions,
            'serverless': GCP_Functions,
            'app engine': GCP_AppEngine,
            'gke': GCP_GKE,
            'filestore': GCP_Filestore,
            'sql': GCP_SQL,
            'spanner': GCP_Spanner,
            'bigtable': GCP_Bigtable,
            'memorystore': GCP_Memorystore,
            'firestore': GCP_Firestore,
            'storage': GCP_Storage,
            'gcs': GCP_GCS,
            'cloud storage': GCP_GCS,
            'google cloud storage': GCP_GCS,
            'bucket': GCP_GCS,
            'cloud bucket': GCP_GCS,
            'google bucket': GCP_GCS,
            'pubsub': GCP_PubSub,
            'cloud pubsub': GCP_PubSub,
            'google cloud pubsub': GCP_PubSub,
            'dataflow': GCP_Dataflow,
            'cloud dataflow': GCP_Dataflow,
            'google dataflow': GCP_Dataflow,
            'load balancer': GCP_LoadBalancing,
            'cloud load balancing': GCP_LoadBalancing,
            'google cloud load balancing': GCP_LoadBalancing,
            'bigquery': GCP_BigQuery,
            'dataproc': GCP_Dataproc,
            'composer': GCP_Composer,
            'build': GCP_Build,
            'container registry': GCP_ContainerRegistry,
            'source repositories': GCP_SourceRepositories,
            'ai platform': GCP_AIPlatform,
            'automl': GCP_Automl,
            'iap': GCP_IAP,
            'security command center': GCP_SecurityCommandCenter,
            'stackdriver': GCP_Monitoring,
            'cloud monitoring': GCP_Monitoring,
            'cloud logging': GCP_Monitoring,
            'logging': GCP_Monitoring,
            'interconnect': GCP_DedicatedInterconnect,
            'cdn': GCP_CDN,
            'cloud cdn': GCP_CDN,
            'endpoints': GCP_Endpoints,
            'api endpoints': GCP_Endpoints,
            'observability': GCP_Monitoring,
            'google observability': GCP_Monitoring,
            # GCP analytics
            'data catalog': GCP_DataCatalog,
            'data fusion': GCP_DataFusion,
            'data lab': GCP_Datalab,
            'data prep': GCP_Dataprep,
            # GCP compute
            'container optimized os': GCP_ContainerOptimizedOS,
            'gpu': GCP_GPU,
            'kubernetes engine': GCP_KubernetesEngine,
            'run': GCP_Run,
            # GCP database
            'datastore': GCP_Datastore,
            'firestore': GCP_Firestore,
            'memorystore': GCP_Memorystore,
            'spanner': GCP_Spanner,
            'sql': GCP_SQL,
            # GCP devtools
            'gradle app engine plugin': GCP_GradleAppEnginePlugin,
            'ide plugins': GCP_IdePlugins,
            'maven app engine plugin': GCP_MavenAppEnginePlugin,
            'scheduler': GCP_Scheduler,
            'sdk': GCP_SDK,
            'tasks': GCP_Tasks,
            'test lab': GCP_TestLab,
            'tools for eclipse': GCP_ToolsForEclipse,
            'tools for powershell': GCP_ToolsForPowershell,
            'tools for visual studio': GCP_ToolsForVisualStudio,
            # GCP iot
            'iot core': GCP_IotCore,
            # GCP migration
            'transfer appliance': GCP_TransferAppliance,
            # GCP ml
            'advanced solutions lab': GCP_AdvancedSolutionsLab,
            'ai hub': GCP_AIHub,
            'aip platform data labeling service': GCP_AIPlatformDataLabelingService,
            'automl natural language': GCP_AutomlNaturalLanguage,
            'automl tables': GCP_AutomlTables,
            'automl translation': GCP_AutomlTranslation,
            'automl video intelligence': GCP_AutomlVideoIntelligence,
            'automl vision': GCP_AutomlVision,
            'automl': GCP_Automl,
            'dialog flow enterprise edition': GCP_DialogFlowEnterpriseEdition,
            'inference api': GCP_InferenceAPI,
            'jobs api': GCP_JobsAPI,
            'natural language api': GCP_NaturalLanguageAPI,
            'recommendations ai': GCP_RecommendationsAI,
            'speech to text': GCP_SpeechToText,
            'text to speech': GCP_TextToSpeech,
            'tpu': GCP_TPU,
            'translation api': GCP_TranslationAPI,
            'video intelligence api': GCP_VideoIntelligenceAPI,
            'vision api': GCP_VisionAPI,
            # GCP network
            'armor': GCP_Armor,
            'cdn': GCP_CDN,
            'dedicated interconnect': GCP_DedicatedInterconnect,
            'dns': GCP_DNS,
            'external ip addresses': GCP_ExternalIpAddresses,
            'firewall rules': GCP_FirewallRules,
            'load balancing': GCP_LoadBalancing,
            'nat': GCP_NAT,
            'network': GCP_Network,
            'partner interconnect': GCP_PartnerInterconnect,
            'premium network tier': GCP_PremiumNetworkTier,
            'router': GCP_Router,
            'routes': GCP_Routes,
            'standard network tier': GCP_StandardNetworkTier,
            'traffic director': GCP_TrafficDirector,
            'virtual private cloud': GCP_VirtualPrivateCloud,
            'vpn': GCP_VPN,
            # GCP nodes
            'iam': GCP_Iam,
            'dns': GCP_DNS,
        }

        # Common nodes that can be used in both AWS and GCP diagrams
        self.common_node_mapping = {
            'server': Server,
            'postgres': PostgreSQL,
            'internet': Internet,
        }
        
    def _validate_provider_specific_icons(self, nodes: List[Dict]) -> List[Dict]:
        """
        Validate and ensure provider-specific icons are used correctly.
        
        Args:
            nodes: List of nodes to validate
            
        Returns:
            List of validated nodes with correct provider-specific icons
            
        This method ensures that:
        1. Only one cloud provider's icons are used
        2. Common services use the correct provider-specific icon
        3. No mixing of AWS and GCP components
        """
        # Common service mappings for each provider
        aws_mappings = {
            'api gateway': 'apigateway',
            'apigateway': 'apigateway',  # Direct mapping for exact match
            'load balancer': 'elb',
            'alb': 'elb',  # Direct mapping for exact match
            'cdn': 'cloudfront',
            'cloudfront': 'cloudfront',  # Direct mapping for exact match
            'message queue': 'sqs',
            'sqs': 'sqs',  # Direct mapping for exact match
            'storage': 's3',
            's3': 's3',  # Direct mapping for exact match
            'database': 'rds',
            'rds': 'rds',  # Direct mapping for exact match
            'monitoring': 'cloudwatch',
            'cloudwatch': 'cloudwatch',  # Direct mapping for exact match
            'logging': 'cloudtrail',
            'cloudtrail': 'cloudtrail',  # Direct mapping for exact match
            'dns': 'route53',
            'route53': 'route53',  # Direct mapping for exact match
            'container service': 'ecs',
            'ecs': 'ecs',  # Direct mapping for exact match
            'serverless': 'lambda',
            'lambda': 'lambda',  # Direct mapping for exact match
            'compute': 'ec2',
            'ec2': 'ec2',  # Direct mapping for exact match
            'kubernetes': 'eks',
            'eks': 'eks',  # Direct mapping for exact match
        }
        
        gcp_mappings = {
            'api gateway': 'gcpapigateway',
            'endpoints': 'gcpapigateway',  # Direct mapping for exact match
            'load balancer': 'load balancing',
            'load balancing': 'load balancing',  # Direct mapping for exact match
            'cdn': 'cdn',
            'message queue': 'pubsub',
            'pubsub': 'pubsub',  # Direct mapping for exact match
            'storage': 'gcs',
            'gcs': 'gcs',  # Direct mapping for exact match
            'database': 'sql',
            'sql': 'sql',  # Direct mapping for exact match
            'monitoring': 'monitoring',
            'logging': 'logging',
            'dns': 'dns',
            'container service': 'gke',
            'gke': 'gke',  # Direct mapping for exact match
            'serverless': 'functions',
            'functions': 'functions',  # Direct mapping for exact match
            'compute': 'compute engine',
            'compute engine': 'compute engine',  # Direct mapping for exact match
            'kubernetes': 'kubernetes engine',
            'kubernetes engine': 'kubernetes engine',  # Direct mapping for exact match
        }

        # Determine which provider is being used
        aws_types = set(self.aws_node_mapping.keys())
        gcp_types = set(self.gcp_node_mapping.keys())
        
        found_aws = any(node['type'].lower() in aws_types for node in nodes)
        found_gcp = any(node['type'].lower() in gcp_types for node in nodes)
        
        if found_aws and found_gcp:
            raise ValueError("Cannot mix AWS and GCP elements in the same diagram. Please use only one cloud provider.")
        
        # Select the appropriate mappings based on the provider
        provider_mappings = aws_mappings if found_aws else gcp_mappings
        provider_node_mapping = self.aws_node_mapping if found_aws else self.gcp_node_mapping
        
        # Validate and correct each node
        corrected_nodes = []
        for node in nodes:
            node_type = node['type'].lower()
            
            # First check for exact matches in provider mappings
            if node_type in provider_mappings:
                node['type'] = provider_mappings[node_type]
                logger.info(f"Corrected node type from '{node_type}' to '{provider_mappings[node_type]}' for exact provider match")
            else:
                # Then check for partial matches in common names
                for common_name, provider_specific in provider_mappings.items():
                    if common_name in node_type:
                        node['type'] = provider_specific
                        logger.info(f"Corrected node type from '{node_type}' to '{provider_specific}' for common service name")
                        break
            
            # Final validation: ensure the type exists in the provider's node mapping
            if node['type'].lower() not in provider_node_mapping:
                logger.warning(f"Node type '{node['type']}' not found in provider mapping, using Server icon")
                node['type'] = 'server'
            
            corrected_nodes.append(node)
        
        return corrected_nodes

    def generate_diagram(self, nodes: List[Dict], connections: List[List[str]], output_path: str, diagram_name: str = "Infrastructure Diagram") -> str:
        """
        Generate an infrastructure diagram.
        
        Args:
            nodes: List of nodes to include in the diagram
            connections: List of connections between nodes
            output_path: Path to save the diagram
            diagram_name: Name of the diagram
            
        Returns:
            Path to the generated diagram file
            
        This method:
        1. Validates the nodes and connections
        2. Creates the diagram structure
        3. Generates the visual representation
        4. Saves the diagram to the specified path
        """
        # Print cool ASCII art banner
        banner = """
    ██████╗  █████╗ ██╗     ███████╗    ██╗  ██╗ ██████╗ ███╗   ███╗███████╗
    ██╔════╝ ██╔══██╗██║     ██╔════╝    ██║  ██║██╔═══██╗████╗ ████║██╔════╝
    ██║  ███╗███████║██║     █████╗      ███████║██║   ██║██╔████╔██║█████╗  
    ██║   ██║██╔══██║██║     ██╔══╝      ██╔══██║██║   ██║██║╚██╔╝██║██╔══╝  
    ╚██████╔╝██║  ██║███████╗███████╗    ██║  ██║╚██████╔╝██║ ╚═╝ ██║███████╗
     ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝    ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝
                                                                            
    ██╗  ██╗ ██████╗ ███╗   ███╗███████╗    ██████╗  █████╗ ████████╗███████╗
    ██║  ██║██╔═══██╗████╗ ████║██╔════╝    ██╔══██╗██╔══██╗╚══██╔══╝██╔════╝
    ███████║██║   ██║██╔████╔██║█████╗      ██║  ██║███████║   ██║   █████╗  
    ██╔══██║██║   ██║██║╚██╔╝██║██╔══╝      ██║  ██║██╔══██║   ██║   ██╔══╝  
    ██║  ██║╚██████╔╝██║ ╚═╝ ██║███████╗    ██████╔╝██║  ██║   ██║   ███████╗
    ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝    ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝
                                                                            
    ███████╗██╗   ██╗███████╗███████╗██╗ ██████╗██╗   ██╗██╗███╗   ██╗███████╗
    ██╔════╝╚██╗ ██╔╝██╔════╝██╔════╝██║██╔════╝██║   ██║██║████╗  ██║██╔════╝
    █████╗   ╚████╔╝ █████╗  █████╗  ██║██║     ██║   ██║██║██╔██╗ ██║█████╗  
    ██╔══╝    ╚██╔╝  ██╔══╝  ██╔══╝  ██║██║     ╚██╗ ██╔╝██║██║╚██╗██║██╔══╝  
    ███████╗   ██║   ███████╗██║     ██║╚██████╗ ╚████╔╝ ██║██║ ╚████║███████╗
    ╚══════╝   ╚═╝   ╚══════╝╚═╝     ╚═╝ ╚═════╝  ╚═══╝  ╚═╝╚═╝  ╚═══╝╚══════╝
        """
        print(banner)
        print("\nAPI Documentation: http://localhost:8000/docs#/\n")

        # Validate and correct provider-specific icons
        nodes = self._validate_provider_specific_icons(nodes)
        
        # Ensure tmp directory exists
        tmp_dir = "tmp"
        os.makedirs(tmp_dir, exist_ok=True)
        
        try:
            # Generate the diagram directly to the output path
            logger.info(f"Generating diagram to: {output_path}")
            with Diagram(
                diagram_name,
                show=True,
                filename=os.path.splitext(output_path)[0],  # Remove .png extension as Diagram adds it
                outformat="png"
            ) as diag:
                self._create_diagram_structure(diag, nodes, connections)
            
            logger.info(f"Generated diagram saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating diagram: {str(e)}")
            raise

    def _create_diagram_structure(self, diag: Diagram, nodes: List[Dict], connections: List[List[str]]):
        """
        Create the structure of the diagram.
        
        Args:
            diag: Diagram object to modify
            nodes: List of nodes to include
            connections: List of connections to create
            
        This method:
        1. Creates clusters for grouped components
        2. Adds nodes to the diagram
        3. Creates connections between nodes
        """
        logger.info(f"Creating nodes: {nodes}")
        logger.info(f"Creating connections: {connections}")
        clusters = {}  # cluster_name -> list of node dicts
        node_objs = {}  # label -> diagram node object
        cloudwatch_label = None
        cloudwatch_node = None

        # Get the appropriate node mapping based on the nodes present
        node_mapping = self._get_node_mapping(nodes)

        # 1. Group nodes by cluster (if any)
        for node in nodes:
            cluster_name = node.get("cluster")
            node_type = node["type"].lower()
            node_label = node["label"]
            if node_type in ["cloudwatch", "monitoring"]:
                cloudwatch_label = node_label
                continue  # Will add CloudWatch separately
            if cluster_name:
                if cluster_name not in clusters:
                    clusters[cluster_name] = []
                clusters[cluster_name].append(node)
            else:
                clusters.setdefault(None, []).append(node)

        # 2. Create all nodes, placing them in clusters if needed
        for cluster_name, cluster_nodes in clusters.items():
            if cluster_name:
                # Create a cluster context for these nodes
                with Cluster(cluster_name):
                    # Sort nodes within cluster by their order
                    cluster_nodes.sort(key=lambda x: x.get('order', float('inf')))
                    for node in cluster_nodes:
                        node_type = node["type"].lower()
                        node_label = node["label"]
                        if node_type in node_mapping:
                            node_obj = node_mapping[node_type](node_label)
                            logger.info(f"Created node in cluster '{cluster_name}': {node_label} ({node_type})")
                        else:
                            node_obj = Server(node_label)
                            logger.warning(f"Unknown node type in cluster: {node_type} for node {node}. Using Server icon.")
                        node_objs[node_label] = node_obj
            else:
                # Nodes not in any cluster
                # Sort nodes by their order
                cluster_nodes.sort(key=lambda x: x.get('order', float('inf')))
                for node in cluster_nodes:
                    node_type = node["type"].lower()
                    node_label = node["label"]
                    if node_type in node_mapping:
                        node_obj = node_mapping[node_type](node_label)
                        logger.info(f"Created node: {node_label} ({node_type})")
                    else:
                        node_obj = Server(node_label)
                        logger.warning(f"Unknown node type: {node_type} for node {node}. Using Server icon.")
                    node_objs[node_label] = node_obj

        # 2b. Add CloudWatch node off to the side, if present
        if cloudwatch_label:
            cloudwatch_node = AWS_Cloudwatch(f"{cloudwatch_label}\n(monitors all services)")
            node_objs[cloudwatch_label] = cloudwatch_node
            logger.info(f"Placed CloudWatch node off to the side: {cloudwatch_label}")

        # 3. Create connections (arrows) between nodes
        # Always use explicit connections if provided
        # Remove any connections to or from CloudWatch
        filtered_connections = [
            [src, dst] for src, dst in connections
            if src != cloudwatch_label and dst != cloudwatch_label
        ] if cloudwatch_label else connections

        if filtered_connections:
            for src, dst in filtered_connections:
                if src in node_objs and dst in node_objs:
                    node_objs[src] >> node_objs[dst]
                    logger.info(f"Created connection: {src} -> {dst}")
                else:
                    logger.warning(f"Connection skipped: {src} -> {dst} (missing node)")
        else:
            # Fallback: If no connections provided, connect nodes in order (left to right)
            all_nodes = []
            for cluster_nodes in clusters.values():
                all_nodes.extend(cluster_nodes)
            # Sort all nodes by their order
            all_nodes.sort(key=lambda x: x.get('order', float('inf')))
            # Connect nodes sequentially
            for i in range(len(all_nodes) - 1):
                src = all_nodes[i]["label"]
                dst = all_nodes[i + 1]["label"]
                if src in node_objs and dst in node_objs:
                    node_objs[src] >> node_objs[dst]
                    logger.info(f"Fallback connection: {src} -> {dst}")

    def _get_node_mapping(self, nodes: List[Dict]) -> Dict:
        """
        Get the appropriate node mapping based on the components.
        
        Args:
            nodes: List of nodes to analyze
            
        Returns:
            Dictionary mapping node types to their icons
            
        This method:
        1. Determines which cloud provider is being used
        2. Returns the appropriate mapping (AWS, GCP, or common)
        """
        aws_types = set(self.aws_node_mapping.keys())
        gcp_types = set(self.gcp_node_mapping.keys())
        
        found_aws = any(node['type'].lower() in aws_types for node in nodes)
        found_gcp = any(node['type'].lower() in gcp_types for node in nodes)
        
        if found_aws and found_gcp:
            raise ValueError("Cannot mix AWS and GCP elements in the same diagram. Please use only one cloud provider.")
        
        if found_aws:
            return {**self.aws_node_mapping, **self.common_node_mapping}
        elif found_gcp:
            return {**self.gcp_node_mapping, **self.common_node_mapping}
        else:
            return self.common_node_mapping

    def _extract_components_with_order(self, description: str) -> List[Dict]:
        """
        Extract components and their order from the description.
        
        Args:
            description: Natural language description of the infrastructure
            
        Returns:
            List of nodes with their order in the description
            
        This method uses the LLM to:
        1. Determine the cloud provider (AWS or GCP)
        2. Extract components with their types
        3. Maintain the order of components as mentioned
        4. Use provider-specific icons for common services
        """
        try:
            prompt = f"""
You are an expert cloud architect. Given an infrastructure description, extract the main components and their types as a JSON list.
For each component, also include its position (order) in the description, where 1 is the first component mentioned.

IMPORTANT ASSUMPTIONS:
1. First, determine if the user is describing an AWS or GCP architecture. Only use elements from the detected provider. Do NOT mix AWS and GCP components in the same diagram. If you detect a mix, raise an error or ask the user to clarify.

2. For common service names, ALWAYS use the provider-specific icon:
   - For AWS diagrams:
     * "API Gateway" → use "apigateway" (AWS_APIGateway)
     * "Load Balancer" → use "elb" or "alb" (AWS_ELB)
     * "CDN" → use "cloudfront" (AWS_CloudFront)
     * "Message Queue" → use "sqs" (AWS_SQS)
     * "Storage" → use "s3" (AWS_S3)
     * "Database" → use "rds" (AWS_RDS)
     * "Monitoring" → use "cloudwatch" (AWS_Cloudwatch)
     * "Logging" → use "cloudtrail" (AWS_Cloudtrail)
     * "DNS" → use "route53" (AWS_Route53)
     * "Container Service" → use "ecs" or "fargate" (AWS_ECS or AWS_Fargate)
     * "Serverless" → use "lambda" (AWS_Lambda)
     * "Compute" → use "ec2" (AWS_EC2)
     * "Kubernetes" → use "eks" (AWS_EKS)

   - For GCP diagrams:
     * "API Gateway" → use "endpoints" (GCP_Endpoints)
     * "Load Balancer" → use "load balancing" (GCP_LoadBalancing)
     * "CDN" → use "cdn" (GCP_CDN)
     * "Message Queue" → use "pubsub" (GCP_PubSub)
     * "Storage" → use "gcs" (GCP_GCS)
     * "Database" → use "sql" or "spanner" (GCP_SQL or GCP_Spanner)
     * "Monitoring" → use "monitoring" (GCP_Monitoring)
     * "Logging" → use "logging" (GCP_Monitoring)
     * "DNS" → use "dns" (GCP_DNS)
     * "Container Service" → use "gke" (GCP_GKE)
     * "Serverless" → use "functions" (GCP_Functions)
     * "Compute" → use "compute engine" (GCP_ComputeEngine)
     * "Kubernetes" → use "kubernetes engine" (GCP_KubernetesEngine)

3. Any service or application component should be represented as EC2 (AWS_EC2) or Compute Engine (GCP_ComputeEngine) by default, unless the description explicitly mentions serverless or container services.

4. Only use Lambda (AWS_Lambda) or Cloud Functions (GCP_Functions) if the description explicitly mentions:
   - "function"
   - "lambda"
   - "serverless"
   - "FaaS"
   - or similar serverless/function concepts

5. If a component is described as a "service" or "application" without specifying serverless/function, use EC2 (AWS_EC2) or Compute Engine (GCP_ComputeEngine).

6. For each component, match the user's term to the most specific cloud service icon available from the detected provider.

7. Do NOT include any elements from the other cloud provider.

Description:
{description}

Respond ONLY with a JSON array, e.g.:
[
  {{"type": "alb", "label": "App Load Balancer", "order": 1}},
  {{"type": "ec2", "label": "Web Server 1", "cluster": "Web Tier", "order": 2}},
  {{"type": "ec2", "label": "Web Server 2", "cluster": "Web Tier", "order": 3}},
  {{"type": "rds", "label": "Database", "cluster": "DB Tier", "order": 4}}
]

IMPORTANT:
- Ensure all components use the correct provider-specific icons
- Maintain the order of components as mentioned in the description
- Include any clustering information if mentioned
- Use the most specific service type available
"""
            response = self.llm_service.llm.generate_content(prompt)
            nodes = json.loads(response.text)
            return self.diagram_generator.generate_diagram(nodes, output_path)
        except Exception as e:
            logger.error(f"Error generating diagram: {str(e)}")
            raise

class DiagramService:
    """Service for generating infrastructure diagrams."""
    
    def __init__(self, llm_service, diagram_generator):
        self.llm_service = llm_service
        self.diagram_generator = diagram_generator
        
    def generate_diagram(self, description: str, output_path: str) -> str:
        """Generate a diagram based on the description."""
        try:
            if self.llm_service.use_mock:
                nodes = self.llm_service.llm.parse_diagram_description(description)
            else:
                nodes = self._extract_components_with_order(description)
            return self.diagram_generator.generate_diagram(nodes, output_path)
        except Exception as e:
            logger.error(f"Error generating diagram: {str(e)}")
            raise