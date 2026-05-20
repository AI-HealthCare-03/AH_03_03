import { auth } from "../firebase";

export type ApiValue =
  | string
  | number
  | boolean
  | null
  | undefined
  | ApiValue[]
  | { [key: string]: ApiValue };

type ApiOptions = Omit<RequestInit, "body"> & {
  body?: BodyInit | Record<string, ApiValue> | null;
  skipAuth?: boolean;
};

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";

export async function apiRequest<T>(path: string, options: ApiOptions = {}): Promise<T> {
  const headers = new Headers(options.headers);
  const isFormData = options.body instanceof FormData;

  if (!isFormData && options.body !== undefined && options.body !== null) {
    headers.set("Content-Type", "application/json");
  }

  if (!options.skipAuth && auth.currentUser) {
    const token = await auth.currentUser.getIdToken();
    headers.set("Authorization", `Bearer ${token}`);
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
    body:
      options.body === undefined || options.body === null || isFormData
        ? (options.body as BodyInit | null | undefined)
        : JSON.stringify(options.body),
  });

  if (!response.ok) {
    let message = `API 요청 실패 (${response.status})`;
    try {
      const errorBody = (await response.json()) as { detail?: string };
      message = errorBody.detail ?? message;
    } catch {
      // Keep the status based fallback message.
    }
    throw new Error(message);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}
