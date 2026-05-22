export type FamilySummary = {
  connected_count: number;
  pending_invite_count: number;
  shared_alert_count: number;
  abnormal_alert_count: number;
};

export type FamilyMember = {
  id: number;
  display_name: string;
  relationship?: string;
  member_role?: string;
  member_status?: string;
};

// Family API integration will be added after the family backend/DB is implemented.

