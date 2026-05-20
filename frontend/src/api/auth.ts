import { apiRequest } from "./client";

export type BackendUser = {
  id: number;
  login_id?: string | null;
  name?: string | null;
  email: string;
  nickname?: string | null;
  phone_number?: string | null;
  birthday?: string | null;
  gender?: string | null;
  role?: string | null;
  is_active?: boolean;
  auth_provider?: string | null;
  has_firebase_uid?: boolean;
  created_at?: string;
  updated_at?: string;
};

export type SignupPayload = {
  login_id: string;
  email: string;
  password: string;
  name: string;
  gender: "MALE" | "FEMALE";
  birth_date: string;
  phone_number: string;
  nickname?: string;
  address?: string;
  sensitive_data_agreed?: boolean;
  marketing_agreed?: boolean;
};

export type LoginPayload = {
  email?: string;
  login_id?: string;
  password: string;
};

export type LoginResponse = {
  access_token: string;
};

export async function signup(payload: SignupPayload): Promise<{ detail: string }> {
  return apiRequest<{ detail: string }>("/auth/signup", {
    method: "POST",
    body: payload,
    skipAuth: true,
  });
}

export async function login(payload: LoginPayload): Promise<LoginResponse> {
  return apiRequest<LoginResponse>("/auth/login", {
    method: "POST",
    body: payload,
    skipAuth: true,
  });
}

export async function getMe(): Promise<BackendUser> {
  return apiRequest<BackendUser>("/users/me");
}

export async function syncFirebaseUser(token?: string): Promise<BackendUser> {
  return apiRequest<BackendUser>("/auth/firebase/sync", {
    method: "POST",
    body: token ? { id_token: token } : {},
  });
}

export async function getFirebaseMe(): Promise<BackendUser> {
  return apiRequest<BackendUser>("/auth/firebase/me");
}
