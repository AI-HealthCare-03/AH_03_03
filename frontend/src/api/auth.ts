import { apiRequest } from "./client";

export type BackendUser = {
  id: number;
  email: string;
  nickname?: string | null;
  role?: string | null;
  is_active?: boolean;
  auth_provider?: string | null;
  has_firebase_uid?: boolean;
};

export async function syncFirebaseUser(token?: string): Promise<BackendUser> {
  return apiRequest<BackendUser>("/auth/firebase/sync", {
    method: "POST",
    body: token ? { token } : {},
  });
}

export async function getFirebaseMe(): Promise<BackendUser> {
  return apiRequest<BackendUser>("/auth/firebase/me");
}
