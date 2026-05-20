import { apiRequest } from "./client";

export async function listChallenges<T>(): Promise<T> {
  return apiRequest<T>("/challenges");
}

export async function listMyChallenges<T>(): Promise<T> {
  return apiRequest<T>("/challenges/my");
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
