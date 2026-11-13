# Training Data

Place the generated training document here with this filename:

```text
azure_aws_cloud_rag_training_20_pages.txt
```

The document used for this project contains Azure and AWS material covering topics such as:

- cloud foundations
- governance
- virtual machines and EC2
- Azure Virtual Network and Amazon VPC
- Azure Storage and Amazon S3
- databases
- identity and security
- containers and Kubernetes
- serverless services
- monitoring and disaster recovery
- cloud cost optimization
- a multi-cloud RAG architecture example

The pipeline in `src/rag_pipeline.py` loads this file, cleans it, splits it into chunks, generates embeddings, stores the vectors in ChromaDB, retrieves Top-K chunks for a question, and sends the retrieved context to a small Hugging Face instruction model.
