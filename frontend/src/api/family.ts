import { apiRequest } from "./client";

export type FamilyRelationType =
  | "SELF"
  | "FATHER"
  | "MOTHER"
  | "SPOUSE"
  | "CHILD"
  | "SIBLING"
  | "GRANDPARENT"
  | "OTHER";

export type FamilyMemberRole = "OWNER" | "MEMBER" | "GUARDIAN" | "DEPENDENT";
export type FamilyStatus = "ACTIVE" | "REMOVED";
export type FamilyMemberStatus = "ACTIVE" | "INVITED" | "PENDING_UNREGISTERED" | "REMOVED";
export type FamilyInviteStatus = "PENDING" | "ACCEPTED" | "DECLINED" | "EXPIRED" | "CANCELED";

export type FamilyGroup = {
  id: number;
  name: string;
  owner_user_id: number;
  status: FamilyStatus;
  created_at: string;
  updated_at: string;
};

export type FamilyMember = {
  id: number;
  family_id: number;
  user_id: number | null;
  display_name: string;
  phone_number: string | null;
  email: string | null;
  relation_type: FamilyRelationType;
  member_role: FamilyMemberRole;
  status: FamilyMemberStatus;
  is_registered: boolean;
  created_at: string;
  updated_at: string;
};

export type FamilyGroupDetail = FamilyGroup & {
  members: FamilyMember[];
};

export type FamilyInvite = {
  id: number;
  family_id: number;
  inviter_user_id: number;
  invitee_user_id: number | null;
  invitee_email: string | null;
  invitee_phone: string | null;
  relation_type: FamilyRelationType;
  member_role: FamilyMemberRole;
  status: FamilyInviteStatus;
  expires_at: string;
  used_at: string | null;
  created_at: string;
  invite_code: string | null;
};

export type FamilySentInvite = Omit<FamilyInvite, "invite_code">;

export type FamilyInvitePreview = {
  invite_id: number;
  family_id: number;
  family_name: string;
  inviter_display_name: string;
  invitee_email: string | null;
  status: FamilyInviteStatus;
  expires_at: string;
};

export type FamilyShareSetting = {
  id: number;
  family_id: number;
  owner_user_id: number;
  viewer_user_id: number;
  share_health_records: boolean;
  share_analysis_results: boolean;
  share_diet_records: boolean;
  share_medications: boolean;
  share_challenges: boolean;
  share_exam_reports: boolean;
  receive_analysis_alerts: boolean;
  receive_abnormal_value_alerts: boolean;
  receive_medication_alerts: boolean;
  created_at: string;
  updated_at: string;
};

export type CreateFamilyGroupPayload = {
  name: string;
};

export type UpdateFamilyGroupPayload = {
  name?: string;
};

export type AddUnregisteredFamilyMemberPayload = {
  display_name: string;
  relation_type: FamilyRelationType;
  phone_number?: string | null;
  email?: string | null;
};

export type CreateFamilyInvitePayload = {
  invitee_email?: string | null;
  invitee_phone?: string | null;
  invitee_user_id?: number | null;
  relation_type: FamilyRelationType;
  member_role?: FamilyMemberRole;
};

export type FamilyShareSettingUpdatePayload = Partial<
  Pick<
    FamilyShareSetting,
    | "share_health_records"
    | "share_analysis_results"
    | "share_diet_records"
    | "share_medications"
    | "share_challenges"
    | "share_exam_reports"
    | "receive_analysis_alerts"
    | "receive_abnormal_value_alerts"
    | "receive_medication_alerts"
  >
>;

export function createFamilyGroup(payload: CreateFamilyGroupPayload): Promise<FamilyGroup> {
  return apiRequest<FamilyGroup>("/family/groups", {
    method: "POST",
    body: payload,
  });
}

export function listFamilyGroups(): Promise<FamilyGroup[]> {
  return apiRequest<FamilyGroup[]>("/family/groups");
}

export function getFamilyGroup(familyId: number): Promise<FamilyGroupDetail> {
  return apiRequest<FamilyGroupDetail>(`/family/groups/${familyId}`);
}

export function updateFamilyGroup(familyId: number, payload: UpdateFamilyGroupPayload): Promise<FamilyGroup> {
  return apiRequest<FamilyGroup>(`/family/groups/${familyId}`, {
    method: "PATCH",
    body: payload,
  });
}

export function deleteFamilyGroup(familyId: number): Promise<void> {
  return apiRequest<void>(`/family/groups/${familyId}`, {
    method: "DELETE",
  });
}

export function listFamilyMembers(familyId: number): Promise<FamilyMember[]> {
  return apiRequest<FamilyMember[]>(`/family/groups/${familyId}/members`);
}

export function addUnregisteredFamilyMember(
  familyId: number,
  payload: AddUnregisteredFamilyMemberPayload,
): Promise<FamilyMember> {
  return apiRequest<FamilyMember>(`/family/groups/${familyId}/members/unregistered`, {
    method: "POST",
    body: payload,
  });
}

export function removeFamilyMember(memberId: number): Promise<void> {
  return apiRequest<void>(`/family/members/${memberId}`, {
    method: "DELETE",
  });
}

export function createFamilyInvite(familyId: number, payload: CreateFamilyInvitePayload): Promise<FamilyInvite> {
  return apiRequest<FamilyInvite>(`/family/groups/${familyId}/invites`, {
    method: "POST",
    body: payload,
  });
}

export function listFamilyGroupInvites(familyId: number): Promise<FamilySentInvite[]> {
  return apiRequest<FamilySentInvite[]>(`/family/groups/${familyId}/invites`);
}

export function listMyFamilyInvites(): Promise<FamilyInvite[]> {
  return apiRequest<FamilyInvite[]>("/family/invites/me");
}

export function acceptFamilyInvite(inviteId: number): Promise<FamilyMember> {
  return apiRequest<FamilyMember>(`/family/invites/${inviteId}/accept`, {
    method: "POST",
  });
}

export function declineFamilyInvite(inviteId: number): Promise<FamilyInvite> {
  return apiRequest<FamilyInvite>(`/family/invites/${inviteId}/decline`, {
    method: "POST",
  });
}

export function acceptFamilyInviteByCode(code: string): Promise<FamilyMember> {
  return apiRequest<FamilyMember>("/family/invites/code/accept", {
    method: "POST",
    body: { code },
  });
}

export function previewFamilyInviteByCode(inviteCode: string): Promise<FamilyInvitePreview> {
  return apiRequest<FamilyInvitePreview>("/family/invites/code/preview", {
    method: "POST",
    body: { invite_code: inviteCode },
  });
}

export function listFamilyShareSettings(): Promise<FamilyShareSetting[]> {
  return apiRequest<FamilyShareSetting[]>("/family/share-settings");
}

export function listFamilyGroupShareSettings(familyId: number): Promise<FamilyShareSetting[]> {
  return apiRequest<FamilyShareSetting[]>(`/family/groups/${familyId}/share-settings`);
}

export function updateFamilyShareSetting(
  settingId: number,
  payload: FamilyShareSettingUpdatePayload,
): Promise<FamilyShareSetting> {
  return apiRequest<FamilyShareSetting>(`/family/share-settings/${settingId}`, {
    method: "PATCH",
    body: payload,
  });
}
