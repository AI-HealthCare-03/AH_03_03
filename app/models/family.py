from enum import StrEnum

from tortoise import fields, models


class FamilyStatus(StrEnum):
    ACTIVE = "ACTIVE"
    REMOVED = "REMOVED"


class FamilyRelationType(StrEnum):
    SELF = "SELF"
    FATHER = "FATHER"
    MOTHER = "MOTHER"
    SPOUSE = "SPOUSE"
    CHILD = "CHILD"
    SIBLING = "SIBLING"
    GRANDPARENT = "GRANDPARENT"
    OTHER = "OTHER"


class FamilyMemberRole(StrEnum):
    OWNER = "OWNER"
    MEMBER = "MEMBER"
    GUARDIAN = "GUARDIAN"
    DEPENDENT = "DEPENDENT"


class FamilyMemberStatus(StrEnum):
    ACTIVE = "ACTIVE"
    INVITED = "INVITED"
    PENDING_UNREGISTERED = "PENDING_UNREGISTERED"
    REMOVED = "REMOVED"


class FamilyInviteStatus(StrEnum):
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    DECLINED = "DECLINED"
    EXPIRED = "EXPIRED"
    CANCELED = "CANCELED"


class Family(models.Model):
    id = fields.BigIntField(primary_key=True)
    name = fields.CharField(max_length=100)
    owner_user = fields.ForeignKeyField("models.User", related_name="owned_families", on_delete=fields.CASCADE)
    status = fields.CharEnumField(enum_type=FamilyStatus, default=FamilyStatus.ACTIVE)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "families"
        indexes = (("owner_user_id",), ("status",))


class FamilyMember(models.Model):
    id = fields.BigIntField(primary_key=True)
    family = fields.ForeignKeyField("models.Family", related_name="members", on_delete=fields.CASCADE)
    user = fields.ForeignKeyField("models.User", related_name="family_members", null=True, on_delete=fields.SET_NULL)
    display_name = fields.CharField(max_length=100)
    phone_number = fields.CharField(max_length=30, null=True)
    email = fields.CharField(max_length=255, null=True)
    relation_type = fields.CharEnumField(enum_type=FamilyRelationType)
    member_role = fields.CharEnumField(enum_type=FamilyMemberRole, default=FamilyMemberRole.MEMBER)
    status = fields.CharEnumField(enum_type=FamilyMemberStatus, default=FamilyMemberStatus.ACTIVE)
    is_registered = fields.BooleanField(default=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "family_members"
        indexes = (
            ("family_id",),
            ("user_id",),
            ("family_id", "user_id"),
            ("family_id", "status"),
            ("email",),
            ("phone_number",),
        )


class FamilyInvite(models.Model):
    id = fields.BigIntField(primary_key=True)
    family = fields.ForeignKeyField("models.Family", related_name="invites", on_delete=fields.CASCADE)
    inviter_user = fields.ForeignKeyField("models.User", related_name="sent_family_invites", on_delete=fields.CASCADE)
    invitee_user = fields.ForeignKeyField(
        "models.User",
        related_name="received_family_invites",
        null=True,
        on_delete=fields.SET_NULL,
    )
    invitee_email = fields.CharField(max_length=255, null=True)
    invitee_phone = fields.CharField(max_length=30, null=True)
    code_hash = fields.CharField(max_length=128, unique=True)
    relation_type = fields.CharEnumField(enum_type=FamilyRelationType)
    member_role = fields.CharEnumField(enum_type=FamilyMemberRole, default=FamilyMemberRole.MEMBER)
    expires_at = fields.DatetimeField()
    used_at = fields.DatetimeField(null=True)
    status = fields.CharEnumField(enum_type=FamilyInviteStatus, default=FamilyInviteStatus.PENDING)
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "family_invites"
        indexes = (
            ("family_id",),
            ("inviter_user_id",),
            ("invitee_user_id",),
            ("invitee_email",),
            ("invitee_phone",),
            ("status",),
            ("expires_at",),
        )


class FamilyShareSetting(models.Model):
    id = fields.BigIntField(primary_key=True)
    family = fields.ForeignKeyField("models.Family", related_name="share_settings", on_delete=fields.CASCADE)
    owner_user = fields.ForeignKeyField("models.User", related_name="family_share_owners", on_delete=fields.CASCADE)
    viewer_user = fields.ForeignKeyField("models.User", related_name="family_share_viewers", on_delete=fields.CASCADE)
    share_health_records = fields.BooleanField(default=False)
    share_analysis_results = fields.BooleanField(default=False)
    share_diet_records = fields.BooleanField(default=False)
    share_medications = fields.BooleanField(default=False)
    share_challenges = fields.BooleanField(default=False)
    share_exam_reports = fields.BooleanField(default=False)
    receive_analysis_alerts = fields.BooleanField(default=False)
    receive_abnormal_value_alerts = fields.BooleanField(default=False)
    receive_medication_alerts = fields.BooleanField(default=False)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "family_share_settings"
        indexes = (
            ("family_id",),
            ("owner_user_id",),
            ("viewer_user_id",),
            ("family_id", "owner_user_id", "viewer_user_id"),
        )
