from enum import StrEnum

from tortoise import fields, models


class RAGDiseaseType(StrEnum):
    DIABETES = "DIABETES"
    OBESITY = "OBESITY"
    DYSLIPIDEMIA = "DYSLIPIDEMIA"
    COMMON = "COMMON"


class RAGSource(models.Model):
    id = fields.BigIntField(primary_key=True)
    source_key = fields.CharField(max_length=100, null=True, unique=True)
    name = fields.CharField(max_length=100)
    organization = fields.CharField(max_length=100, null=True)
    source_type = fields.CharField(max_length=30)
    base_url = fields.CharField(max_length=500, null=True)
    description = fields.TextField(null=True)
    is_active = fields.BooleanField(default=True)
    metadata = fields.JSONField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "rag_sources"


class RAGDocument(models.Model):
    id = fields.BigIntField(primary_key=True)
    source = fields.ForeignKeyField("models.RAGSource", related_name="documents")
    document_key = fields.CharField(max_length=200, null=True, unique=True)
    source_key = fields.CharField(max_length=100, null=True)
    title = fields.CharField(max_length=255)
    disease_type = fields.CharEnumField(enum_type=RAGDiseaseType)
    disease_code = fields.CharField(max_length=50, null=True)
    filename = fields.CharField(max_length=255, null=True)
    document_url = fields.CharField(max_length=500, null=True)
    review_status = fields.CharField(max_length=50, null=True)
    usage_scope = fields.TextField(null=True)
    published_at = fields.DateField(null=True)
    fetched_at = fields.DatetimeField(null=True)
    version = fields.CharField(max_length=50, null=True)
    is_active = fields.BooleanField(default=True)
    metadata = fields.JSONField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "rag_documents"
        indexes = (
            ("source_id",),
            ("disease_type", "is_active"),
            ("source_key",),
            ("disease_code", "is_active"),
            ("review_status", "is_active"),
        )


class RAGChunk(models.Model):
    id = fields.BigIntField(primary_key=True)
    document = fields.ForeignKeyField("models.RAGDocument", related_name="chunks")
    chunk_key = fields.CharField(max_length=200, null=True, unique=True)
    chunk_index = fields.IntField()
    section_title = fields.CharField(max_length=255, null=True)
    content = fields.TextField()
    content_hash = fields.CharField(max_length=64, null=True)
    content_length = fields.IntField(null=True)
    token_estimate = fields.IntField(null=True)
    disease_type = fields.CharEnumField(enum_type=RAGDiseaseType)
    keywords = fields.TextField(null=True)
    embedding_model = fields.CharField(max_length=100, null=True)
    embedding_provider = fields.CharField(max_length=100, null=True)
    embedding_dimension = fields.IntField(null=True)
    embedding_content_hash = fields.CharField(max_length=64, null=True)
    embedded_at = fields.DatetimeField(null=True)
    is_active = fields.BooleanField(default=True)
    metadata = fields.JSONField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "rag_chunks"
        indexes = (
            ("document_id", "chunk_index"),
            ("disease_type",),
            ("content_hash",),
            ("is_active",),
        )


class RAGRetrievalLog(models.Model):
    id = fields.BigIntField(primary_key=True)
    user = fields.ForeignKeyField(
        "models.User", related_name="rag_retrieval_logs", null=True, on_delete=fields.SET_NULL
    )
    analysis_result = fields.ForeignKeyField(
        "models.AnalysisResult", related_name="rag_retrieval_logs", null=True, on_delete=fields.SET_NULL
    )
    query_text = fields.TextField()
    disease_type = fields.CharEnumField(enum_type=RAGDiseaseType, null=True)
    retrieved_chunk_ids = fields.JSONField(null=True)
    top_k = fields.IntField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "rag_retrieval_logs"
        indexes = (("user_id", "created_at"), ("analysis_result_id",))
