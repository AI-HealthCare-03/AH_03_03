import { apiRequest } from "./client";

export async function listChallenges<T>(params: { category?: string; limit?: number; offset?: number } = {}): Promise<T> {
  const query = new URLSearchParams();
  if (params.category) {
    query.set("category", params.category);
  }
  if (params.limit !== undefined) {
    query.set("limit", String(params.limit));
  }
  if (params.offset !== undefined) {
    query.set("offset", String(params.offset));
  }
  const suffix = query.toString() ? `?${query.toString()}` : "";
  return apiRequest<T>(`/challenges${suffix}`);
}

export async function getChallenge<T>(challengeId: number): Promise<T> {
  return apiRequest<T>(`/challenges/${challengeId}`, { skipAuth: true });
}

export async function listMyChallenges<T>(params: { limit?: number; offset?: number } = {}): Promise<T> {
  const query = new URLSearchParams();
  if (params.limit !== undefined) {
    query.set("limit", String(params.limit));
  }
  if (params.offset !== undefined) {
    query.set("offset", String(params.offset));
  }
  const suffix = query.toString() ? `?${query.toString()}` : "";
  return apiRequest<T>(`/challenges/my${suffix}`);
}

export async function listChallengeLogs<T>(userChallengeId: number): Promise<T> {
  return apiRequest<T>(`/challenges/my/${userChallengeId}/logs`);
}

export async function joinChallenge<T>(challengeId: number): Promise<T> {
  return apiRequest<T>(`/challenges/${challengeId}/join`, { method: "POST" });
}

export async function completeToday<T>(userChallengeId: number): Promise<T> {
  return apiRequest<T>(`/challenges/my/${userChallengeId}/complete-today`, { method: "POST" });
}

export async function giveUpChallenge<T>(userChallengeId: number): Promise<T> {
  return apiRequest<T>(`/challenges/my/${userChallengeId}/give-up`, { method: "PATCH" });
}

export async function listChallengeRecommendations<T>(params: { limit?: number; offset?: number } = {}): Promise<T> {
  const query = new URLSearchParams();
  if (params.limit !== undefined) {
    query.set("limit", String(params.limit));
  }
  if (params.offset !== undefined) {
    query.set("offset", String(params.offset));
  }
  const suffix = query.toString() ? `?${query.toString()}` : "";
  return apiRequest<T>(`/challenges/recommendations${suffix}`);
}
