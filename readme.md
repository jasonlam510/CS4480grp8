# Northern Lights Video Analysis

This is the group project of the course CS4480 Data Intensive Computing at CityUHK.

## Description

This project demonstrates how to set up and run a **distributed Hadoop Big Data ecosystem on a single PC** using Docker Compose. Instead of requiring multiple physical machines, we simulate a distributed cluster with multiple containers, each running different Hadoop services. This approach allows students to learn and experiment with Big Data technologies (HDFS, HBase, YARN, Spark) without needing expensive hardware or cloud infrastructure.

The project implements a 3-stage pipeline for analyzing Northern Lights videos:
1. **Stage 1**: Video retrieval and download from YouTube
2. **Stage 2**: Video preprocessing and feature extraction
3. **Stage 3**: Distributed model training using PySpark on YARN

All data is stored in HDFS (distributed file system) and metadata in HBase (NoSQL database), with processing handled by Spark running on YARN (resource manager).

## 1. Hadoop Environment

### 1.1 Hadoop Components

Our Docker Compose setup includes the following Hadoop ecosystem components:

#### **HDFS (Hadoop Distributed File System)** - Storage Layer
- **NameNode**: Manages the file system namespace and regulates access to files. Acts as the central coordinator for HDFS.
- **DataNode1 & DataNode2**: Store actual data blocks. Two DataNodes enable data replication (replication factor: 2) for fault tolerance.

#### **YARN (Yet Another Resource Negotiator)** - Resource Management
- **ResourceManager**: Manages cluster resources and schedules applications. Coordinates with NodeManagers to allocate resources for jobs.
- **NodeManager1 & NodeManager2**: Run on worker nodes, manage containers, and monitor resource usage. These execute Spark applications submitted to YARN.

#### **HBase** - NoSQL Database for Metadata
- **HBase Master**: Manages the HBase cluster, coordinates RegionServers, and handles table operations.
- **HBase RegionServer**: Handles read/write requests and manages table regions. Stores metadata (video IDs, timestamps, tags, etc.).
- **Zookeeper**: Required coordination service for HBase. Manages cluster state and configuration.

#### **Spark** - Distributed Processing
- **Spark Client**: Container for submitting PySpark jobs to YARN. Jobs run as YARN applications, not in standalone mode.

### 1.2 Resource Allocation

The cluster is designed to run on a machine with **8 CPU cores and 16GB RAM**. Resource limits are configured for each service to ensure stable operation.

#### Total Resource Usage
- **RAM**: ~12.5GB allocated (leaves 3.5GB for OS and Docker overhead)
- **CPU**: ~6.25 cores allocated (leaves 1.75 cores for OS and Docker overhead)

#### Node-by-Node Resource Allocation

| Component | Nodes | RAM per Node | CPU per Node | Total RAM | Total CPU | Purpose |
|-----------|-------|--------------|--------------|-----------|-----------|---------|
| **NameNode** | 1 | 1GB | 0.5 cores | 1GB | 0.5 cores | HDFS metadata management |
| **DataNode** | 2 | 1GB each | 0.5 cores each | 2GB | 1.0 core | HDFS data storage (with replication) |
| **ResourceManager** | 1 | 1GB | 0.5 cores | 1GB | 0.5 cores | YARN resource scheduling |
| **NodeManager** | 2 | 2GB each | 1.0 core each | 4GB | 2.0 cores | YARN workers (run Spark executors) |
| **HBase Master** | 1 | 1GB | 0.5 cores | 1GB | 0.5 cores | HBase cluster management |
| **HBase RegionServer** | 1 | 2GB | 1.0 core | 2GB | 1.0 core | HBase data storage/query |
| **Zookeeper** | 1 | 512MB | 0.25 cores | 512MB | 0.25 cores | Coordination service |
| **Spark Client** | 1 | 1GB | 0.5 cores | 1GB | 0.5 cores | Job submission container |
| **TOTAL** | **10 containers** | - | - | **~12.5GB** | **~6.25 cores** | - |

#### Storage Allocation

- **Docker Volumes**: Each service has persistent storage volumes that grow dynamically based on actual data:
  - NameNode metadata volume
  - 2 DataNode volumes (for data blocks)
  - 2 NodeManager volumes (for local YARN storage)
  
- **HDFS Storage**: 
  - Data is distributed across 2 DataNodes
  - Replication factor: 2 (each block stored on 2 DataNodes)
  - Actual disk usage = data size × replication factor
  
- **No hard storage limits**: Volumes grow as needed. Monitor with `docker system df -v`

#### Available Resources for Spark Jobs

When submitting Spark jobs to YARN, the following resources are available:
- **RAM**: ~4GB (from 2 NodeManagers)
- **CPU**: 2 cores (from 2 NodeManagers)
- Example: Can run 4 executors with `--executor-memory 1g --executor-cores 1`

For detailed resource allocation information, see [docs/RESOURCE_ALLOCATION.md](docs/RESOURCE_ALLOCATION.md).

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- At least 8GB RAM available for containers

### Setup

1. **Start the Hadoop cluster:**
   ```bash
   docker compose up -d
   ```

2. **Wait for services to be ready** (about 1-2 minutes), then initialize HDFS and HBase:
   ```bash
   ./scripts/setup.sh
   ```

3. **Verify the setup:**
   - HDFS NameNode Web UI: http://localhost:9870
   - YARN ResourceManager: http://localhost:8088
   - HBase Master: http://localhost:16010

4. **Run the pipeline stages:**
   - Stage 1: `./scripts/run_stage1.sh` (download videos)
   - Stage 2: `./scripts/run_stage2.sh` (preprocess videos)
   - Stage 3: `./scripts/run_stage3.sh` (train model on YARN)

### Environment Variables

Create a `.env` file in the project root with:
```
YOUTUBE_API_KEY=your_api_key_here
```

### Stopping the Cluster

```bash
docker compose down
```

To remove all data volumes:
```bash
docker compose down -v
```

## Project Structure

See [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for detailed directory structure and organization.

## Documentation

- [PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) - Complete project structure and organization
- [RESOURCE_ALLOCATION.md](docs/RESOURCE_ALLOCATION.md) - Detailed resource allocation and monitoring guide

## Technologies

- **Storage**: HDFS, HBase
- **Processing**: PySpark, YARN
- **Infrastructure**: Docker, Docker Compose
- **Analysis**: R, ggplot2
- **Languages**: Python, R, Shell