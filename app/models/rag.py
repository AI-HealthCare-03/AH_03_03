from enum import StrEnum

from tortoise import fields, models


class RAGDiseaseType(StrEnum):
    DIABETES = "DIABETES"
    OBESITY = "OBESITY"
    DYSLIPIDEMIA = "DYSLIPIDEMIA"
    COMMON = "COMMON"


class RAGSource(models.Model):
    id = fields.BigIntField(primary_key=True)
    name = fields.CharField(max_length=100)
    organization = fields.CharField(max_length=100, null=True)
    source_type = fields.CharField(max_length=30)
    base_url = fields.CharField(max_length=500, null=True)
    description = fields.TextField(null=True)
    is_active = fields.BooleanField(default=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "rag_sources"


class RAGDocument(models.Model):
    id = fields.BigIntField(primary_key=True)
    source = fields.ForeignKeyField("models.RAGSource", related_name="documents")
    title = fields.CharField(max_length=255)
    disease_type = fields.CharEnumField(enum_type=RAGDiseaseType)
    document_url = fields.CharField(max_length=500, null=True)
    published_at = fields.DateField(null=True)
    fetched_at = fields.DatetimeField(null=True)
    version = fields.CharField(max_length=50, null=True)
    is_active = fields.BooleanField(default=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "rag_documents"
        indexes = (("source_id",), ("disease_type", "is_active"))


class RAGChunk(models.Model):
    id = fields.BigIntField(primary_key=True)
    document = fields.ForeignKeyField("models.RAGDocument", related_name="chunks")
    chunk_index = fields.IntField()
    section_title = fields.CharField(max_length=255, null=True)
    content = fields.TextField()
    disease_type = fields.CharEnumField(enum_type=RAGDiseaseType)
    keywords = fields.TextField(null=True)
    embedding_model = fields.CharField(max_length=100, null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "rag_chunks"
        indexes = (("document_id", "chunk_index"), ("disease_type",))


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
