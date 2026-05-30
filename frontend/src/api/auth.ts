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
  phone_number?: string;
  nickname?: string;
  address?: string;
  privacy_consent_agreed?: boolean;
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

export type PasswordResetRequestResponse = {
  detail: string;
  debug_token?: string | null;
};

export type PasswordResetConfirmPayload = {
  token: string;
  new_password: string;
};

export type AvailabilityResponse = {
  available: boolean;
  message?: string | null;
};

export type EmailVerificationSendResponse = {
  detail: string;
  debug_code?: string | null;
};

export type EmailVerificationVerifyResponse = {
  verified: boolean;
};

export type FindLoginIdPayload = {
  name: string;
  email?: string;
  phone_number?: string;
};

export type FindLoginIdResponse = {
  found: boolean;
  masked_login_id: string | null;
  message: string;
};

export type UserUpdatePayload = {
  name?: string;
  login_id?: string;
  nickname?: string;
  email?: string;
  phone_number?: string;
  birthday?: string;
  gender?: "MALE" | "FEMALE";
  address?: string;
  profile_image_url?: string;
};

export type PasswordChangePayload = {
  current_password: string;
  new_password: string;
};

export async function signup(payload: SignupPayload): Promise<{ detail: string }> {
  return apiRequest<{ detail: string }>("/auth/signup", {
    method: "POST",
    body: payload,
    skipAuth: true,
  });
}

export async function checkLoginId(loginId: string): Promise<AvailabilityResponse> {
  const params = new URLSearchParams({ login_id: loginId });
  return apiRequest<AvailabilityResponse>(`/auth/check-login-id?${params.toString()}`, {
    skipAuth: true,
  });
}

export async function checkEmail(email: string): Promise<AvailabilityResponse> {
  const params = new URLSearchParams({ email });
  return apiRequest<AvailabilityResponse>(`/auth/check-email?${params.toString()}`, {
    skipAuth: true,
  });
}

export async function checkPhone(phoneNumber: string): Promise<AvailabilityResponse> {
  const params = new URLSearchParams({ phone_number: phoneNumber });
  return apiRequest<AvailabilityResponse>(`/auth/check-phone?${params.toString()}`, {
    skipAuth: true,
  });
}

export async function sendEmailVerification(email: string): Promise<EmailVerificationSendResponse> {
  return apiRequest<EmailVerificationSendResponse>("/auth/email-verifications/send", {
    method: "POST",
    body: { email },
    skipAuth: true,
  });
}

export async function verifyEmailCode(email: string, code: string): Promise<EmailVerificationVerifyResponse> {
  return apiRequest<EmailVerificationVerifyResponse>("/auth/email-verifications/verify", {
    method: "POST",
    body: { email, code },
    skipAuth: true,
  });
}

export async function findLoginId(payload: FindLoginIdPayload): Promise<FindLoginIdResponse> {
  return apiRequest<FindLoginIdResponse>("/auth/find-login-id", {
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

export async function updateMe(payload: UserUpdatePayload): Promise<BackendUser> {
  return apiRequest<BackendUser>("/users/me", {
    method: "PATCH",
    body: payload,
  });
}

export async function deactivateMe(): Promise<BackendUser> {
  return apiRequest<BackendUser>("/users/me", {
    method: "DELETE",
  });
}

export async function changePassword(payload: PasswordChangePayload): Promise<{ detail: string }> {
  return apiRequest<{ detail: string }>("/auth/password", {
    method: "PATCH",
    body: payload,
  });
}

export async function logout(): Promise<{ detail: string }> {
  return apiRequest<{ detail: string }>("/auth/logout", {
    method: "POST",
    skipRefresh: true,
  });
}

export async function requestPasswordReset(email: string): Promise<PasswordResetRequestResponse> {
  return apiRequest<PasswordResetRequestResponse>("/auth/password-reset/request", {
    method: "POST",
    body: { email },
    skipAuth: true,
  });
}

export async function confirmPasswordReset(payload: PasswordResetConfirmPayload): Promise<{ detail: string }> {
  return apiRequest<{ detail: string }>("/auth/password-reset/confirm", {
    method: "POST",
    body: payload,
    skipAuth: true,
  });
}
